"""
backend/services/ncf_service.py
=================================
Single authoritative NCF inference service.

Responsibilities
----------------
1. Detect whether a trained NCF model exists on disk.
2. Load the model exactly once (module-level singleton).
3. Provide predict_for_user(user_id, food_ids) → List[Optional[float]]
   where each element is a genuine model score in [0, 1], or None when
   the model is unavailable / user or food is unknown.
4. Expose is_available() so callers can branch without catching exceptions.
5. Report the current NCF status in a dict (for UI and logging).

What this service NEVER does
------------------------------
- Generate random scores.
- Generate sine-hash pseudo-scores.
- Return a fake constant when the model is absent.
- Accept health parameters (diseases, severity, BMI) as input — those
  belong to the nutrition/content scoring pipeline, not NCF.
- Silently pretend NCF is active when no model file exists.

Model paths
-----------
Primary:   ncf_integration/models/ncf_model.keras    (Keras native format)
Mappings:  ncf_integration/models/ncf_mappings.json  (user→idx, food_id→idx)

ncf_mappings.json schema
-------------------------
{
  "user_to_idx":  {"alice": 0, "bob": 1, ...},
  "food_id_to_idx": {"1": 0, "2": 1, ..., "1014": 1013},
  "num_users": <int>,
  "num_items": <int>,
  "trained_on": "<ISO-8601 timestamp>",
  "training_interactions": <int>
}

When training is run (ncf_integration/models/train_ncf.py), the trainer
saves both the .keras model and ncf_mappings.json at the same time.
The mapping file is the authoritative source of which user strings and
food_ids the model was trained on.

Unknown user / unknown food handling
--------------------------------------
If a user_id string is not in user_to_idx → return None (not fake score).
If a food_id integer is not in food_id_to_idx → return None (not fake score).
The hybrid scorer in enhanced_recommender.py treats None as NCF unavailable
for that specific (user, food) pair and falls back to nutrition+content only.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Model paths (relative to project root) ──────────────────────────────────
_MODEL_PATH    = os.path.join("ncf_integration", "models", "ncf_model.keras")
_MAPPINGS_PATH = os.path.join("ncf_integration", "models", "ncf_mappings.json")

# ── Output scale: SimpleNCF uses linear output + MSE on normalised ratings.
# Ratings were normalised as (rating - 1) / 4  →  raw output ∈ ~[0, 1].
# We clamp and treat the raw output directly as a [0, 1] preference score.
_OUTPUT_SCALE = 1.0   # no re-scaling needed; just clamp to [0, 1]


# ────────────────────────────────────────────────────────────────────────────
#  Singleton state  (module-level — loaded once per Python process)
# ────────────────────────────────────────────────────────────────────────────

_model         = None          # loaded keras Model or None
_user_to_idx:  Dict[str, int] = {}
_food_id_to_idx: Dict[int, int] = {}
_num_users:    int = 0
_num_items:    int = 0
_trained_on:   Optional[str] = None
_training_interactions: int = 0
_load_attempted: bool = False  # set True after first load attempt
_load_error:   Optional[str] = None


def _attempt_load() -> None:
    """
    Try to load the NCF model and mappings from disk.

    Sets module-level state.  Called at most once per process (guarded by
    _load_attempted).  Safe to call from multiple threads (Python GIL
    protects simple assignments).
    """
    global _model, _user_to_idx, _food_id_to_idx
    global _num_users, _num_items, _trained_on, _training_interactions
    global _load_attempted, _load_error

    _load_attempted = True

    # ── Check model file ──────────────────────────────────────────────
    if not os.path.isfile(_MODEL_PATH):
        _load_error = f"Model file not found: {_MODEL_PATH}"
        logger.info("NCF: %s", _load_error)
        return

    if os.path.getsize(_MODEL_PATH) == 0:
        _load_error = f"Model file is empty: {_MODEL_PATH}"
        logger.warning("NCF: %s", _load_error)
        return

    # ── Check mappings file ───────────────────────────────────────────
    if not os.path.isfile(_MAPPINGS_PATH):
        _load_error = f"Mappings file not found: {_MAPPINGS_PATH}"
        logger.warning("NCF: %s", _load_error)
        return

    try:
        with open(_MAPPINGS_PATH, encoding="utf-8") as fh:
            mappings = json.load(fh)
    except Exception as exc:
        _load_error = f"Cannot parse mappings file: {exc}"
        logger.warning("NCF: %s", _load_error)
        return

    # ── Validate mappings schema ──────────────────────────────────────
    required_keys = {"user_to_idx", "food_id_to_idx", "num_users", "num_items"}
    missing = required_keys - set(mappings.keys())
    if missing:
        _load_error = f"Mappings file missing keys: {missing}"
        logger.warning("NCF: %s", _load_error)
        return

    # ── Load Keras model ──────────────────────────────────────────────
    try:
        import tensorflow as tf
        loaded = tf.keras.models.load_model(_MODEL_PATH, compile=False)
    except Exception as exc:
        _load_error = f"Keras model load failed: {exc}"
        logger.warning("NCF: %s", _load_error)
        return

    # ── Verify expected layer names ───────────────────────────────────
    layer_names = {layer.name for layer in loaded.layers}
    if "user_embedding" not in layer_names or "item_embedding" not in layer_names:
        _load_error = (
            f"Loaded model is missing expected layers "
            f"('user_embedding', 'item_embedding'). "
            f"Found: {sorted(layer_names)}"
        )
        logger.warning("NCF: %s", _load_error)
        return

    # ── All checks passed — commit state ─────────────────────────────
    _model = loaded

    # user_to_idx keys are strings (JSON), values are ints
    _user_to_idx = {str(k): int(v) for k, v in mappings["user_to_idx"].items()}

    # food_id_to_idx keys stored as strings in JSON; convert to int for lookup
    _food_id_to_idx = {int(k): int(v) for k, v in mappings["food_id_to_idx"].items()}

    _num_users   = int(mappings["num_users"])
    _num_items   = int(mappings["num_items"])
    _trained_on  = mappings.get("trained_on")
    _training_interactions = int(mappings.get("training_interactions", 0))
    _load_error  = None

    logger.info(
        "NCF model loaded: %d users, %d items, trained_on=%s",
        _num_users, _num_items, _trained_on,
    )


# ── Trigger load at import time ──────────────────────────────────────────────
_attempt_load()


# ────────────────────────────────────────────────────────────────────────────
#  Public API
# ────────────────────────────────────────────────────────────────────────────

def is_available() -> bool:
    """
    Return True if and only if a trained NCF model is loaded and ready.

    Never returns True when no model was found on disk.
    Never returns True when mappings are missing or corrupt.
    Never returns True when the loaded model has incorrect architecture.
    """
    return _model is not None


def get_status() -> Dict:
    """
    Return a status dict for logging, debugging, and the scoring_info block.

    Keys:
        available          bool   — True iff model is loaded
        model_path         str    — path checked
        mappings_path      str    — path checked
        num_users          int    — users in the trained model (0 if unavailable)
        num_items          int    — items in the trained model (0 if unavailable)
        trained_on         str|None — ISO-8601 training timestamp
        training_interactions int — number of interactions model was trained on
        load_error         str|None — human-readable reason if unavailable
    """
    return {
        "available":              is_available(),
        "model_path":             _MODEL_PATH,
        "mappings_path":          _MAPPINGS_PATH,
        "num_users":              _num_users,
        "num_items":              _num_items,
        "trained_on":             _trained_on,
        "training_interactions":  _training_interactions,
        "load_error":             _load_error,
    }


def predict_for_user(
    user_id: str,
    food_ids: List[int],
) -> List[Optional[float]]:
    """
    Return NCF preference scores for a user over a list of food_ids.

    Each score is a genuine model prediction in [0, 1], or None when:
    - The model is not loaded (no trained model on disk).
    - user_id was not seen during training (unknown user).
    - A food_id was not seen during training (unknown food).
    - The model raises an exception during inference.

    The hybrid scorer treats None as "NCF unavailable for this pair" and
    falls back to nutrition + content scoring for that food.

    Args:
        user_id:  Application username string (same as stored in food_interactions).
        food_ids: List of integer food_ids (from food_dataset.csv).

    Returns:
        List of float|None, same length as food_ids.
        Order is preserved: result[i] corresponds to food_ids[i].

    NEVER returns fake/random/constant scores.
    """
    if not is_available():
        return [None] * len(food_ids)

    if not food_ids:
        return []

    # ── Resolve user index ────────────────────────────────────────────
    user_str = str(user_id).strip()
    user_idx = _user_to_idx.get(user_str)
    if user_idx is None:
        logger.debug(
            "NCF: unknown user '%s' — returning None for all %d foods",
            user_str, len(food_ids),
        )
        return [None] * len(food_ids)

    # ── Resolve food indices — track which positions are valid ────────
    valid_positions: List[int] = []
    valid_food_indices: List[int] = []

    results: List[Optional[float]] = [None] * len(food_ids)

    for pos, fid in enumerate(food_ids):
        fidx = _food_id_to_idx.get(int(fid))
        if fidx is None:
            # Unknown food — leave results[pos] = None
            logger.debug("NCF: unknown food_id %s — skipping", fid)
        else:
            valid_positions.append(pos)
            valid_food_indices.append(fidx)

    if not valid_positions:
        return results

    # ── Batch prediction ──────────────────────────────────────────────
    try:
        user_array = np.full(len(valid_food_indices), user_idx, dtype=np.int32)
        item_array = np.array(valid_food_indices, dtype=np.int32)

        raw_preds = _model.predict(
            [user_array, item_array],
            verbose=0,
            batch_size=min(256, len(valid_food_indices)),
        )
        raw_preds = raw_preds.flatten()

        for i, pos in enumerate(valid_positions):
            raw = float(raw_preds[i])
            # Clamp to [0, 1] — SimpleNCF uses linear output; training targets
            # were normalised to [0,1] but inference may produce slight outliers
            score = max(0.0, min(1.0, raw))
            results[pos] = score

    except Exception as exc:
        logger.warning(
            "NCF prediction failed for user '%s': %s. "
            "Returning None for all foods.",
            user_str, exc,
        )
        return [None] * len(food_ids)

    return results


def predict_single(user_id: str, food_id: int) -> Optional[float]:
    """
    Convenience wrapper for a single (user, food) pair.

    Returns a float in [0, 1] or None (never fake).
    """
    results = predict_for_user(user_id, [food_id])
    return results[0] if results else None


def reload() -> bool:
    """
    Force reload of the NCF model and mappings from disk.

    Useful after training completes in the same process.
    Returns True if reload succeeded and model is now available.
    """
    global _load_attempted
    _load_attempted = False
    # Reset state
    global _model, _user_to_idx, _food_id_to_idx
    global _num_users, _num_items, _trained_on, _training_interactions, _load_error
    _model = None
    _user_to_idx = {}
    _food_id_to_idx = {}
    _num_users = _num_items = _training_interactions = 0
    _trained_on = _load_error = None

    _attempt_load()
    return is_available()
