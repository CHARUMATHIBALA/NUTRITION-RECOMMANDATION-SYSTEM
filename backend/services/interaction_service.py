"""
backend/services/interaction_service.py
=========================================
Real user-food interaction recording and retrieval service.

This module is the ONLY place in the project that writes to the
food_interactions table.  All interaction recording goes through the
public functions here so the schema and validation rules stay in one place.

Interaction schema
------------------
user_id          TEXT    — application username string
food_id          INTEGER — food_dataset.csv food_id (1-indexed, 1–1014)
interaction_type TEXT    — 'selected' | 'swapped_to' | 'swapped_away'
timestamp        TEXT    — ISO-8601 UTC datetime

What constitutes a genuine interaction
---------------------------------------
'selected'    — The food appeared in the finalised meal plan that was
                shown to the user after running the analysis.  This is a
                passive-positive signal (the system recommended it and
                the user did not reject it).

'swapped_to'  — The user explicitly chose this food via the swap UI to
                replace another food.  This is a strong-positive signal.

'swapped_away'— The user replaced this food with something else via the
                swap UI.  This is a weak-negative signal.

These are the only three interaction types.  Disease labels, severity
levels, BMI, and other health parameters are NOT interactions and are
NEVER stored as NCF training signals.

NCF training gate
-----------------
The module exports MIN_INTERACTIONS_FOR_TRAINING.  The NCF training
pipeline must check this before attempting to train; training on fewer
interactions than this threshold is not meaningful and is blocked.

Currently: MIN_INTERACTIONS_FOR_TRAINING = 500 unique (user, food) pairs
with at least 50 distinct users and 100 distinct foods.  These minimums
are conservative — real CF models benefit from much larger datasets.

Data integrity rules
---------------------
1. food_id must be in range [1, 1014] (matches food_dataset.csv).
2. user_id must be a non-empty string.
3. interaction_type must be one of the three valid types.
4. Duplicate (user_id, food_id, interaction_type) on the same calendar day
   are silently ignored — one interaction per food per type per day per user.
5. No personal health data (diseases, severity, BMI, etc.) is stored here.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from backend.database import _get_conn

logger = logging.getLogger(__name__)

# ── Interaction types ────────────────────────────────────────────────────────
INTERACTION_SELECTED    = "selected"
INTERACTION_SWAPPED_TO  = "swapped_to"
INTERACTION_SWAPPED_AWAY = "swapped_away"

VALID_INTERACTION_TYPES = {
    INTERACTION_SELECTED,
    INTERACTION_SWAPPED_TO,
    INTERACTION_SWAPPED_AWAY,
}

# ── food_id range from food_dataset.csv (1-indexed, 1014 foods) ─────────────
FOOD_ID_MIN = 1
FOOD_ID_MAX = 1014

# ── NCF training readiness thresholds ────────────────────────────────────────
# Minimum number of unique (user_id, food_id) positive-interaction pairs
# required before NCF training is considered meaningful.
MIN_INTERACTIONS_FOR_TRAINING  = 500
MIN_USERS_FOR_TRAINING         = 50
MIN_FOODS_FOR_TRAINING         = 100


# ────────────────────────────────────────────────────────────────────────────
#  Validation
# ────────────────────────────────────────────────────────────────────────────

def _validate_interaction(
    user_id: str,
    food_id: int,
    interaction_type: str,
) -> Optional[str]:
    """
    Return an error string if the interaction is invalid, else None.

    Checks:
    - user_id is a non-empty string
    - food_id is an integer in [FOOD_ID_MIN, FOOD_ID_MAX]
    - interaction_type is one of VALID_INTERACTION_TYPES
    """
    if not isinstance(user_id, str) or not user_id.strip():
        return f"Invalid user_id: must be a non-empty string, got {user_id!r}"

    try:
        fid = int(food_id)
    except (TypeError, ValueError):
        return f"Invalid food_id: must be an integer, got {food_id!r}"

    if fid < FOOD_ID_MIN or fid > FOOD_ID_MAX:
        return (
            f"Invalid food_id {fid}: must be in [{FOOD_ID_MIN}, {FOOD_ID_MAX}] "
            f"(matches food_dataset.csv range)"
        )

    if interaction_type not in VALID_INTERACTION_TYPES:
        return (
            f"Invalid interaction_type {interaction_type!r}: "
            f"must be one of {sorted(VALID_INTERACTION_TYPES)}"
        )

    return None   # valid


# ────────────────────────────────────────────────────────────────────────────
#  Recording API
# ────────────────────────────────────────────────────────────────────────────

def record_interaction(
    user_id: str,
    food_id: int,
    interaction_type: str,
    timestamp: Optional[str] = None,
) -> bool:
    """
    Record a single real user-food interaction.

    This is the primary write API.  Call it whenever a genuine interaction
    event occurs in the application:
    - When a meal plan is shown to the user → interaction_type='selected'
      for each food in the plan.
    - When the user clicks "swap" and accepts a new food → 'swapped_to'
      for the new food, 'swapped_away' for the replaced food.

    Duplicate (user_id, food_id, interaction_type) records on the same
    UTC calendar day are silently ignored.

    Args:
        user_id:          Application username string.
        food_id:          food_dataset.csv food_id (integer, 1–1014).
        interaction_type: One of 'selected', 'swapped_to', 'swapped_away'.
        timestamp:        ISO-8601 UTC string; defaults to now.

    Returns:
        True if the record was inserted, False if validation failed or
        the insert was skipped (duplicate).
    """
    error = _validate_interaction(user_id, food_id, interaction_type)
    if error:
        logger.warning("Interaction rejected: %s", error)
        return False

    ts = timestamp or datetime.now(timezone.utc).isoformat()
    # Extract just the date portion for daily deduplication
    date_prefix = ts[:10]   # e.g. "2025-09-06"

    try:
        conn = _get_conn()
        cur  = conn.cursor()
        # Insert only if no identical (user, food, type) record exists today
        cur.execute(
            """
            INSERT INTO food_interactions (user_id, food_id, interaction_type, timestamp)
            SELECT ?, ?, ?, ?
            WHERE NOT EXISTS (
                SELECT 1 FROM food_interactions
                WHERE user_id          = ?
                  AND food_id          = ?
                  AND interaction_type = ?
                  AND substr(timestamp, 1, 10) = ?
            )
            """,
            (
                user_id.strip(), int(food_id), interaction_type, ts,
                user_id.strip(), int(food_id), interaction_type, date_prefix,
            ),
        )
        inserted = cur.rowcount > 0
        conn.commit()
        conn.close()

        if inserted:
            logger.debug(
                "Interaction recorded: user=%s food_id=%s type=%s",
                user_id, food_id, interaction_type,
            )
        return inserted

    except Exception:
        logger.exception(
            "Failed to record interaction: user=%s food_id=%s type=%s",
            user_id, food_id, interaction_type,
        )
        return False


def record_meal_plan_selections(
    user_id: str,
    meal_plan: Dict[str, List],
) -> int:
    """
    Record 'selected' interactions for every food in a generated meal plan.

    This is the primary hook for passive interaction collection.  Call it
    immediately after a meal plan is shown to the user.

    Args:
        user_id:   Application username string.
        meal_plan: Dict[meal_type → list of food dicts or pd.Series].
                   Each food item must have a 'food_id' key/field.

    Returns:
        Number of interactions successfully recorded.
    """
    import pandas as pd

    recorded = 0
    for meal_type, foods in meal_plan.items():
        for food in foods:
            # Support both dict and pd.Series
            if isinstance(food, pd.Series):
                fid = food.get("food_id")
            elif isinstance(food, dict):
                fid = food.get("food_id")
            else:
                continue

            if fid is None:
                continue

            ok = record_interaction(user_id, fid, INTERACTION_SELECTED)
            if ok:
                recorded += 1

    logger.info(
        "Meal plan interactions recorded: user=%s foods=%d", user_id, recorded
    )
    return recorded


def record_swap(
    user_id: str,
    old_food_id: int,
    new_food_id: int,
) -> bool:
    """
    Record a swap interaction: 'swapped_away' for the old food,
    'swapped_to' for the new food.

    Args:
        user_id:     Application username string.
        old_food_id: food_id of the food being replaced.
        new_food_id: food_id of the food being chosen.

    Returns:
        True if both records were inserted successfully.
    """
    away_ok = record_interaction(user_id, old_food_id, INTERACTION_SWAPPED_AWAY)
    to_ok   = record_interaction(user_id, new_food_id, INTERACTION_SWAPPED_TO)
    return away_ok and to_ok


# ────────────────────────────────────────────────────────────────────────────
#  Retrieval API
# ────────────────────────────────────────────────────────────────────────────

def get_interactions_for_user(user_id: str, limit: int = 1000) -> List[Dict]:
    """
    Return all interactions for a specific user, newest first.

    Args:
        user_id: Application username string.
        limit:   Maximum number of records to return.

    Returns:
        List of dicts with keys: id, user_id, food_id, interaction_type, timestamp.
    """
    try:
        conn = _get_conn()
        cur  = conn.cursor()
        cur.execute(
            """
            SELECT id, user_id, food_id, interaction_type, timestamp
            FROM food_interactions
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (user_id.strip(), limit),
        )
        rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        logger.exception("Failed to retrieve interactions for user=%s", user_id)
        return []


def get_all_interactions(
    interaction_types: Optional[List[str]] = None,
    limit: int = 100_000,
) -> List[Dict]:
    """
    Return interactions across all users, optionally filtered by type.

    Used by the NCF training pipeline to build the training dataset.
    Only 'selected' and 'swapped_to' are positive interactions; callers
    should filter appropriately before building training tensors.

    Args:
        interaction_types: List of types to include; None = all types.
        limit:             Maximum records to return.

    Returns:
        List of dicts with keys: user_id, food_id, interaction_type, timestamp.
    """
    try:
        conn = _get_conn()
        cur  = conn.cursor()

        if interaction_types:
            placeholders = ",".join("?" * len(interaction_types))
            cur.execute(
                f"""
                SELECT user_id, food_id, interaction_type, timestamp
                FROM food_interactions
                WHERE interaction_type IN ({placeholders})
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (*interaction_types, limit),
            )
        else:
            cur.execute(
                """
                SELECT user_id, food_id, interaction_type, timestamp
                FROM food_interactions
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (limit,),
            )

        rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        logger.exception("Failed to retrieve all interactions")
        return []


# ────────────────────────────────────────────────────────────────────────────
#  Training readiness check
# ────────────────────────────────────────────────────────────────────────────

def get_interaction_stats() -> Dict:
    """
    Return statistics about the current interaction dataset.

    Used by the NCF training gate to decide whether training is feasible.

    Returns dict with keys:
        total_interactions     — total rows in food_interactions
        unique_user_food_pairs — unique (user_id, food_id) combinations
        unique_users           — distinct user_id values
        unique_foods           — distinct food_id values
        positive_interactions  — 'selected' + 'swapped_to' count
        negative_interactions  — 'swapped_away' count
        training_ready         — bool: all minimums met
        reason                 — human-readable explanation of status
    """
    try:
        conn = _get_conn()
        cur  = conn.cursor()

        total = cur.execute(
            "SELECT COUNT(*) FROM food_interactions"
        ).fetchone()[0]

        unique_pairs = cur.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT user_id, food_id FROM food_interactions)"
        ).fetchone()[0]

        unique_users = cur.execute(
            "SELECT COUNT(DISTINCT user_id) FROM food_interactions"
        ).fetchone()[0]

        unique_foods = cur.execute(
            "SELECT COUNT(DISTINCT food_id) FROM food_interactions"
        ).fetchone()[0]

        positive = cur.execute(
            "SELECT COUNT(*) FROM food_interactions "
            "WHERE interaction_type IN ('selected','swapped_to')"
        ).fetchone()[0]

        negative = cur.execute(
            "SELECT COUNT(*) FROM food_interactions "
            "WHERE interaction_type = 'swapped_away'"
        ).fetchone()[0]

        conn.close()

        # Evaluate training readiness
        ready = (
            unique_pairs >= MIN_INTERACTIONS_FOR_TRAINING
            and unique_users >= MIN_USERS_FOR_TRAINING
            and unique_foods >= MIN_FOODS_FOR_TRAINING
        )

        if ready:
            reason = (
                f"Training ready: {unique_pairs} unique pairs, "
                f"{unique_users} users, {unique_foods} foods."
            )
        else:
            shortfalls = []
            if unique_pairs < MIN_INTERACTIONS_FOR_TRAINING:
                shortfalls.append(
                    f"need {MIN_INTERACTIONS_FOR_TRAINING} unique (user,food) pairs "
                    f"(have {unique_pairs})"
                )
            if unique_users < MIN_USERS_FOR_TRAINING:
                shortfalls.append(
                    f"need {MIN_USERS_FOR_TRAINING} distinct users "
                    f"(have {unique_users})"
                )
            if unique_foods < MIN_FOODS_FOR_TRAINING:
                shortfalls.append(
                    f"need {MIN_FOODS_FOR_TRAINING} distinct foods "
                    f"(have {unique_foods})"
                )
            reason = (
                "NCF NOT TRAINED — insufficient real interaction data. "
                + "; ".join(shortfalls) + ". "
                "Record more real user-food interactions before training."
            )

        return {
            "total_interactions":     total,
            "unique_user_food_pairs": unique_pairs,
            "unique_users":           unique_users,
            "unique_foods":           unique_foods,
            "positive_interactions":  positive,
            "negative_interactions":  negative,
            "training_ready":         ready,
            "reason":                 reason,
        }

    except Exception:
        logger.exception("Failed to get interaction stats")
        return {
            "total_interactions":     0,
            "unique_user_food_pairs": 0,
            "unique_users":           0,
            "unique_foods":           0,
            "positive_interactions":  0,
            "negative_interactions":  0,
            "training_ready":         False,
            "reason":                 "Error querying interaction database.",
        }
