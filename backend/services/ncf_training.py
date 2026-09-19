"""
backend/services/ncf_training.py
==================================
NeuMF (Neural Matrix Factorisation) training pipeline for the Smart
Nutrition Analyzer.

Architecture: He et al., 2017 — "Neural Collaborative Filtering"
  GMF path : element-wise product of user and item embeddings
  MLP path : concatenated embeddings through dense hidden layers
  Combined : concatenate GMF and MLP outputs → Dense(1, sigmoid)

Loss      : binary_crossentropy (implicit feedback binarised at threshold)
Optimiser : Adam (lr from config.NCF_LEARNING_RATE)

Data contract
-------------
The training CSV must contain columns:
  user_id  — integer 0-based user index
  food_id  — integer matching food_dataset.csv food_id (1–1014)
  rating   — float 1.0–5.0

Binarisation rule (config.NCF_POSITIVE_THRESHOLD = 3.5):
  rating >= threshold → label 1 (positive / preferred)
  rating <  threshold → label 0 (negative / not preferred)

Negative sampling
-----------------
For every positive (user, food) pair we sample NCF_NEGATIVE_RATIO
food_ids that the user has NOT interacted with. Sampling is seeded
(NCF_RANDOM_SEED) for reproducibility and only picks food_ids that
exist in food_dataset.csv (1–1014).

Artifacts saved
---------------
  ncf_integration/models/ncf_model.keras      — trained Keras model
  ncf_integration/models/ncf_mappings.json    — user/food index maps
  ncf_integration/models/training_metadata.json — full provenance

Usage
-----
  from backend.services.ncf_training import train_ncf
  result = train_ncf()          # uses config defaults + CSV path
  print(result['val_loss'])
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Project-root-relative imports ────────────────────────────────────────────
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config

# ── Constants ─────────────────────────────────────────────────────────────────
_MODEL_PATH    = os.path.join(_ROOT, config.NCF_MODEL_PATH)
_MAPPINGS_PATH = os.path.join(_ROOT, config.NCF_MAPPINGS_PATH)
_METADATA_PATH = os.path.join(_ROOT, config.NCF_METADATA_PATH)
_INTERACTIONS_CSV = os.path.join(_ROOT, config.NCF_INTERACTIONS_CSV)
_FOOD_CSV         = os.path.join(_ROOT, "food_dataset.csv")


# ─────────────────────────────────────────────────────────────────────────────
#  1. Data loading & preparation
# ─────────────────────────────────────────────────────────────────────────────

def _load_valid_food_ids() -> List[int]:
    """Return the set of valid food_ids from food_dataset.csv (1–1014)."""
    df = pd.read_csv(_FOOD_CSV)
    return sorted(df["food_id"].astype(int).tolist())


def _load_interactions(csv_path: str) -> pd.DataFrame:
    """
    Load interaction CSV and validate schema.

    Expected columns: user_id, food_id, rating
    Returns a clean DataFrame with int user_id, int food_id, float rating.
    """
    df = pd.read_csv(csv_path)
    required = {"user_id", "food_id", "rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Interaction CSV missing columns: {missing}")
    df["user_id"] = df["user_id"].astype(int)
    df["food_id"] = df["food_id"].astype(int)
    df["rating"]  = df["rating"].astype(float)
    logger.info("Loaded %d interaction rows from %s", len(df), csv_path)
    return df


def _build_mappings(
    df: pd.DataFrame,
    valid_food_ids: List[int],
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, int]]:
    """
    Build dense integer index mappings.

    Returns
    -------
    user_to_idx   : original user_id  → dense index
    idx_to_user   : dense index → original user_id
    food_to_idx   : food_id (1–1014)  → dense index
    idx_to_food   : dense index → food_id
    """
    # Users: sorted unique user_ids in the interaction data
    unique_users = sorted(df["user_id"].unique().tolist())
    user_to_idx  = {u: i for i, u in enumerate(unique_users)}
    idx_to_user  = {i: u for u, i in user_to_idx.items()}

    # Foods: ALL valid food_ids from food_dataset.csv (ensures full coverage)
    food_to_idx = {fid: idx for idx, fid in enumerate(valid_food_ids)}
    idx_to_food = {idx: fid for fid, idx in food_to_idx.items()}

    logger.info(
        "Mappings built: %d users, %d foods", len(user_to_idx), len(food_to_idx)
    )
    return user_to_idx, idx_to_user, food_to_idx, idx_to_food


def _binarise(
    df: pd.DataFrame,
    threshold: float,
    user_to_idx: Dict[int, int],
    food_to_idx: Dict[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert continuous ratings to binary labels and apply index mapping.

    Only keeps rows where food_id is in food_to_idx (valid food_dataset foods).

    Returns
    -------
    user_indices  : shape (N,) int32
    item_indices  : shape (N,) int32
    labels        : shape (N,) float32  — 1.0 if rating >= threshold else 0.0
    """
    # Filter to known foods only (safety: drop synthetic food_ids not in dataset)
    df = df[df["food_id"].isin(food_to_idx)].copy()

    user_enc  = df["user_id"].map(user_to_idx).values.astype(np.int32)
    item_enc  = df["food_id"].map(food_to_idx).values.astype(np.int32)
    labels    = (df["rating"] >= threshold).astype(np.float32).values

    logger.info(
        "Binarised: %d total rows, %d positive (%.1f%%)",
        len(labels),
        int(labels.sum()),
        100 * labels.mean(),
    )
    return user_enc, item_enc, labels


def _negative_sample(
    user_enc: np.ndarray,
    item_enc: np.ndarray,
    labels: np.ndarray,
    num_items: int,
    ratio: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Add deterministic negative samples for positive interactions.

    For every row where label == 1 (positive), sample `ratio` item indices
    that the same user has NOT interacted with (across the whole dataset).

    Rules
    -----
    - Only sample negatives for positive interactions (label == 1).
    - Never treat an existing positive as a negative.
    - Seeded with `seed` for reproducibility.
    - Only uses indices in [0, num_items).
    """
    rng = np.random.default_rng(seed)

    # Build per-user set of ALL interacted item indices (positive and negative)
    user_interacted: dict[int, set] = {}
    for u, i in zip(user_enc, item_enc):
        user_interacted.setdefault(int(u), set()).add(int(i))

    pos_mask = labels == 1.0
    pos_users = user_enc[pos_mask]
    pos_items = item_enc[pos_mask]

    all_items = np.arange(num_items, dtype=np.int32)

    neg_users: list = []
    neg_items: list = []

    for u, _ in zip(pos_users, pos_items):
        u_int = int(u)
        interacted = user_interacted.get(u_int, set())
        # candidate negatives = all items NOT interacted by this user
        candidates = np.array(
            [i for i in all_items if i not in interacted], dtype=np.int32
        )
        if len(candidates) == 0:
            continue
        n_samples = min(ratio, len(candidates))
        chosen = rng.choice(candidates, size=n_samples, replace=False)
        neg_users.extend([u_int] * n_samples)
        neg_items.extend(chosen.tolist())

    neg_users_arr = np.array(neg_users, dtype=np.int32)
    neg_items_arr = np.array(neg_items, dtype=np.int32)
    neg_labels    = np.zeros(len(neg_users_arr), dtype=np.float32)

    combined_users  = np.concatenate([user_enc,  neg_users_arr])
    combined_items  = np.concatenate([item_enc,  neg_items_arr])
    combined_labels = np.concatenate([labels,    neg_labels])

    # Shuffle combined dataset with the same seed (deterministic)
    idx = rng.permutation(len(combined_labels))
    combined_users  = combined_users[idx]
    combined_items  = combined_items[idx]
    combined_labels = combined_labels[idx]

    logger.info(
        "After negative sampling: %d total (%d positive, %d negative)",
        len(combined_labels),
        int(combined_labels.sum()),
        int((combined_labels == 0).sum()),
    )
    return combined_users, combined_items, combined_labels


# ─────────────────────────────────────────────────────────────────────────────
#  2. NeuMF model construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_neumf(
    num_users: int,
    num_items: int,
    embedding_dim: int,
    mlp_layers: List[int],
    dropout_rate: float,
    learning_rate: float,
    seed: int,
) -> "tf.keras.Model":
    """
    Build a NeuMF model (He et al., 2017).

    Architecture
    ------------
    GMF path
        user_embedding_gmf (num_users, embedding_dim)
        item_embedding_gmf (num_items, embedding_dim)
        element-wise multiply → gmf_out (embedding_dim,)

    MLP path
        user_embedding_mlp (num_users, embedding_dim)
        item_embedding_mlp (num_items, embedding_dim)
        concatenate → (2 * embedding_dim,)
        Dense(mlp_layers[0], relu) → Dropout
        Dense(mlp_layers[1], relu)
        → mlp_out (mlp_layers[-1],)

    Combine
        concatenate(gmf_out, mlp_out) → Dense(32, relu) → Dense(1, sigmoid)

    Both embedding tables use glorot_uniform initialisation with the
    same seed so training is reproducible.
    """
    import tensorflow as tf

    tf.random.set_seed(seed)
    np.random.seed(seed)

    initialiser = tf.keras.initializers.GlorotUniform(seed=seed)

    # ── Inputs ────────────────────────────────────────────────────────
    user_input = tf.keras.Input(shape=(1,), dtype="int32", name="user_input")
    item_input = tf.keras.Input(shape=(1,), dtype="int32", name="item_input")

    # ── GMF path ──────────────────────────────────────────────────────
    user_emb_gmf = tf.keras.layers.Embedding(
        num_users, embedding_dim,
        embeddings_initializer=initialiser,
        name="user_embedding",   # ← must match ncf_service check
    )(user_input)
    item_emb_gmf = tf.keras.layers.Embedding(
        num_items, embedding_dim,
        embeddings_initializer=initialiser,
        name="item_embedding",   # ← must match ncf_service check
    )(item_input)

    user_flat_gmf = tf.keras.layers.Flatten(name="user_flat_gmf")(user_emb_gmf)
    item_flat_gmf = tf.keras.layers.Flatten(name="item_flat_gmf")(item_emb_gmf)

    gmf_out = tf.keras.layers.Multiply(name="gmf_multiply")([user_flat_gmf, item_flat_gmf])

    # ── MLP path ──────────────────────────────────────────────────────
    user_emb_mlp = tf.keras.layers.Embedding(
        num_users, embedding_dim,
        embeddings_initializer=initialiser,
        name="user_embedding_mlp",
    )(user_input)
    item_emb_mlp = tf.keras.layers.Embedding(
        num_items, embedding_dim,
        embeddings_initializer=initialiser,
        name="item_embedding_mlp",
    )(item_input)

    user_flat_mlp = tf.keras.layers.Flatten(name="user_flat_mlp")(user_emb_mlp)
    item_flat_mlp = tf.keras.layers.Flatten(name="item_flat_mlp")(item_emb_mlp)

    mlp_vec = tf.keras.layers.Concatenate(name="mlp_concat")(
        [user_flat_mlp, item_flat_mlp]
    )
    for i, units in enumerate(mlp_layers):
        mlp_vec = tf.keras.layers.Dense(
            units, activation="relu", name=f"mlp_dense_{i}"
        )(mlp_vec)
        mlp_vec = tf.keras.layers.Dropout(
            dropout_rate, name=f"mlp_dropout_{i}", seed=seed
        )(mlp_vec)

    # ── Combine GMF + MLP ─────────────────────────────────────────────
    combined = tf.keras.layers.Concatenate(name="gmf_mlp_concat")([gmf_out, mlp_vec])
    combined = tf.keras.layers.Dense(32, activation="relu", name="combine_dense")(combined)
    output   = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(combined)

    model = tf.keras.Model(
        inputs=[user_input, item_input],
        outputs=output,
        name="NeuMF",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  3. Train / validation split
# ─────────────────────────────────────────────────────────────────────────────

def _split(
    users: np.ndarray,
    items: np.ndarray,
    labels: np.ndarray,
    val_fraction: float,
    seed: int,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
]:
    """
    Deterministic train / validation split (no data leakage — split is by row
    index after the full dataset has been shuffled once during negative sampling).
    """
    from sklearn.model_selection import train_test_split  # already a dependency
    (
        u_tr, u_va,
        i_tr, i_va,
        l_tr, l_va,
    ) = train_test_split(
        users, items, labels,
        test_size=val_fraction,
        random_state=seed,
        stratify=labels,
    )
    logger.info(
        "Split: train=%d  val=%d", len(l_tr), len(l_va)
    )
    return u_tr, i_tr, l_tr, u_va, i_va, l_va


# ─────────────────────────────────────────────────────────────────────────────
#  4. Save artifacts
# ─────────────────────────────────────────────────────────────────────────────

def _save_mappings(
    user_to_idx: Dict[int, int],
    food_to_idx: Dict[int, int],
    num_users: int,
    num_items: int,
    num_interactions: int,
) -> None:
    """Save ncf_mappings.json consumed by ncf_service.py."""
    os.makedirs(os.path.dirname(_MAPPINGS_PATH), exist_ok=True)
    mappings = {
        # Keys are strings (JSON) matching user_id as-used in the application.
        # During training on synthetic data, user_ids are integers 0–999.
        # In production, these will be real username strings from the DB.
        "user_to_idx":     {str(k): int(v) for k, v in user_to_idx.items()},
        "food_id_to_idx":  {str(k): int(v) for k, v in food_to_idx.items()},
        "num_users":        num_users,
        "num_items":        num_items,
        "trained_on":       datetime.now(timezone.utc).isoformat(),
        "training_interactions": num_interactions,
    }
    with open(_MAPPINGS_PATH, "w", encoding="utf-8") as fh:
        json.dump(mappings, fh, indent=2)
    logger.info("Mappings saved → %s", _MAPPINGS_PATH)


def _save_metadata(meta: dict) -> None:
    """Save full training provenance to training_metadata.json."""
    os.makedirs(os.path.dirname(_METADATA_PATH), exist_ok=True)
    with open(_METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    logger.info("Metadata saved → %s", _METADATA_PATH)


# ─────────────────────────────────────────────────────────────────────────────
#  5. Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def train_ncf(
    interactions_csv: Optional[str] = None,
    *,
    embedding_dim:    int   = None,
    mlp_layers:       list  = None,
    dropout_rate:     float = None,
    learning_rate:    float = None,
    epochs:           int   = None,
    batch_size:       int   = None,
    negative_ratio:   int   = None,
    positive_threshold: float = None,
    val_fraction:     float = 0.10,
    seed:             int   = None,
    verbose:          int   = 1,
) -> Dict:
    """
    Train a NeuMF model and save all artifacts.

    All numeric parameters default to values in config.py when not supplied.

    Returns
    -------
    dict with keys:
        success          bool
        num_users        int
        num_items        int
        num_interactions int
        positive_pairs   int
        negative_pairs   int
        train_samples    int
        val_samples      int
        epochs_run       int
        val_loss         float
        val_accuracy     float
        val_auc          float
        model_path       str
        mappings_path    str
        metadata_path    str
        error            str | None
    """
    # ── Resolve parameters from config ──────────────────────────────
    csv_path          = interactions_csv or _INTERACTIONS_CSV
    embedding_dim     = embedding_dim    or config.NCF_EMBEDDING_DIM
    mlp_layers        = mlp_layers       or list(config.NCF_MLP_LAYERS)
    dropout_rate      = dropout_rate     if dropout_rate is not None else config.NCF_DROPOUT_RATE
    learning_rate     = learning_rate    if learning_rate is not None else config.NCF_LEARNING_RATE
    epochs            = epochs           or config.NCF_EPOCHS
    batch_size        = batch_size       or config.NCF_BATCH_SIZE
    negative_ratio    = negative_ratio   if negative_ratio is not None else config.NCF_NEGATIVE_RATIO
    positive_threshold = positive_threshold if positive_threshold is not None else config.NCF_POSITIVE_THRESHOLD
    seed              = seed             if seed is not None else config.NCF_RANDOM_SEED

    result: Dict = {
        "success": False, "num_users": 0, "num_items": 0,
        "num_interactions": 0, "positive_pairs": 0, "negative_pairs": 0,
        "train_samples": 0, "val_samples": 0, "epochs_run": 0,
        "val_loss": None, "val_accuracy": None, "val_auc": None,
        "model_path": _MODEL_PATH,
        "mappings_path": _MAPPINGS_PATH,
        "metadata_path": _METADATA_PATH,
        "error": None,
    }

    t_start = time.time()

    try:
        # ── Import TF inside function (graceful if TF missing) ───────
        import tensorflow as tf
        tf.random.set_seed(seed)
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        tf.get_logger().setLevel("WARNING")

        # ── Load data ────────────────────────────────────────────────
        print("[NCF] Loading interaction data …")
        df = _load_interactions(csv_path)
        valid_food_ids = _load_valid_food_ids()

        user_to_idx, _, food_to_idx, _ = _build_mappings(df, valid_food_ids)
        num_users = len(user_to_idx)
        num_items = len(food_to_idx)   # = 1014

        result["num_users"]        = num_users
        result["num_items"]        = num_items
        result["num_interactions"] = len(df)

        # ── Binarise ────────────────────────────────────────────────
        print("[NCF] Binarising ratings at threshold %.1f …" % positive_threshold)
        u_enc, i_enc, labels = _binarise(df, positive_threshold, user_to_idx, food_to_idx)
        positive_pairs = int(labels.sum())
        result["positive_pairs"] = positive_pairs

        # ── Negative sampling ────────────────────────────────────────
        print("[NCF] Generating %d× negative samples …" % negative_ratio)
        u_all, i_all, l_all = _negative_sample(
            u_enc, i_enc, labels, num_items, negative_ratio, seed
        )
        negative_pairs = int((l_all == 0).sum())
        result["negative_pairs"] = negative_pairs

        # ── Train / val split ────────────────────────────────────────
        print("[NCF] Splitting train / val …")
        u_tr, i_tr, l_tr, u_va, i_va, l_va = _split(
            u_all, i_all, l_all, val_fraction, seed
        )
        result["train_samples"] = len(l_tr)
        result["val_samples"]   = len(l_va)

        # ── Build model ──────────────────────────────────────────────
        print("[NCF] Building NeuMF (%d users, %d items, emb=%d) …" % (
            num_users, num_items, embedding_dim
        ))
        model = _build_neumf(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            mlp_layers=mlp_layers,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            seed=seed,
        )
        if verbose:
            model.summary(print_fn=lambda x: print("[NCF]   " + x))

        # ── Callbacks ───────────────────────────────────────────────
        os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=_MODEL_PATH,
                monitor="val_loss",
                save_best_only=True,
                mode="min",
                verbose=0,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.NCF_EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                mode="min",
                verbose=1,
            ),
        ]

        # ── Train ─────────────────────────────────────────────────
        print("[NCF] Training (max %d epochs, early_stop patience=%d) …" % (
            epochs, config.NCF_EARLY_STOPPING_PATIENCE
        ))
        history = model.fit(
            [u_tr, i_tr], l_tr,
            validation_data=([u_va, i_va], l_va),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )

        epochs_run = len(history.history["loss"])
        val_loss   = float(min(history.history["val_loss"]))
        val_acc    = float(history.history["val_accuracy"][
            history.history["val_loss"].index(min(history.history["val_loss"]))
        ])
        val_auc    = float(history.history.get("val_auc", [0.0])[
            history.history["val_loss"].index(min(history.history["val_loss"]))
        ])

        result.update({
            "epochs_run":   epochs_run,
            "val_loss":     val_loss,
            "val_accuracy": val_acc,
            "val_auc":      val_auc,
        })

        # ── Save mappings ─────────────────────────────────────────
        print("[NCF] Saving mappings …")
        _save_mappings(
            user_to_idx, food_to_idx,
            num_users, num_items, len(df),
        )

        # ── Save metadata ─────────────────────────────────────────
        elapsed = round(time.time() - t_start, 1)
        metadata = {
            "architecture":        "NeuMF (GMF + MLP)",
            "embedding_dim":       embedding_dim,
            "mlp_layers":          mlp_layers,
            "dropout_rate":        dropout_rate,
            "learning_rate":       learning_rate,
            "batch_size":          batch_size,
            "max_epochs":          epochs,
            "epochs_run":          epochs_run,
            "early_stopping_patience": config.NCF_EARLY_STOPPING_PATIENCE,
            "negative_ratio":      negative_ratio,
            "positive_threshold":  positive_threshold,
            "val_fraction":        val_fraction,
            "random_seed":         seed,
            "loss_function":       "binary_crossentropy",
            "num_users":           num_users,
            "num_items":           num_items,
            "num_interactions":    len(df),
            "positive_pairs":      positive_pairs,
            "negative_pairs":      negative_pairs,
            "train_samples":       len(l_tr),
            "val_samples":         len(l_va),
            "val_loss":            val_loss,
            "val_accuracy":        val_acc,
            "val_auc":             val_auc,
            "training_date":       datetime.now(timezone.utc).isoformat(),
            "training_time_s":     elapsed,
            "data_source":         csv_path,
            "food_dataset_path":   _FOOD_CSV,
            "model_path":          _MODEL_PATH,
            "mappings_path":       _MAPPINGS_PATH,
            "data_note": (
                "Trained on synthetic interaction data. "
                "No real user food_interactions exist yet. "
                "Retrain when >= 500 real (user, food) pairs are available."
            ),
        }
        _save_metadata(metadata)

        result["success"] = True
        print(
            "[NCF] Training complete. "
            "val_loss=%.4f  val_accuracy=%.4f  val_auc=%.4f  time=%.1fs" % (
                val_loss, val_acc, val_auc, elapsed
            )
        )

    except Exception as exc:
        logger.exception("NCF training failed: %s", exc)
        result["error"] = str(exc)
        print("[NCF] Training FAILED: %s" % exc)

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    res = train_ncf(verbose=1)
    if res["success"]:
        print("\n[NCF] SUCCESS")
        print("  Users     :", res["num_users"])
        print("  Items     :", res["num_items"])
        print("  Positives :", res["positive_pairs"])
        print("  Negatives :", res["negative_pairs"])
        print("  Train     :", res["train_samples"])
        print("  Val       :", res["val_samples"])
        print("  Epochs    :", res["epochs_run"])
        print("  val_loss  :", round(res["val_loss"], 4))
        print("  val_acc   :", round(res["val_accuracy"], 4))
        print("  val_auc   :", round(res["val_auc"], 4))
    else:
        print("\n[NCF] FAILED:", res["error"])
