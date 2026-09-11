from dataclasses import dataclass
from typing import Optional


@dataclass
class ValidationConfig:
    """Configuration for weekly plan validation checks.

    All limits are inclusive. If a limit is set to None, that check is disabled.
    """
    # Daily nutrition limits (min, max) for each metric
    calories_min: Optional[float] = None
    calories_max: Optional[float] = None
    protein_min: Optional[float] = None
    protein_max: Optional[float] = None
    carbs_min: Optional[float] = None
    carbs_max: Optional[float] = None
    fats_min: Optional[float] = None
    fats_max: Optional[float] = None
    fiber_min: Optional[float] = None
    fiber_max: Optional[float] = None
    sodium_min: Optional[float] = None
    sodium_max: Optional[float] = None

    # Tolerance for per‑meal calorie distribution (percentage of target)
    calorie_distribution_tolerance: float = 0.05  # 5%

    # Minimum distinct food categories per day
    min_distinct_categories: int = 3

    # Flag to enable or disable optional checks
    enable_optional_checks: bool = True


def load_default_validation_config() -> ValidationConfig:
    """Return a ValidationConfig with sensible defaults.

    These defaults are deliberately permissive and can be overridden by the
    caller (e.g., tests) to enforce stricter constraints.
    """
    return ValidationConfig(
        calories_min=1500,
        calories_max=2500,
        protein_min=50,
        protein_max=200,
        carbs_min=150,
        carbs_max=350,
        fats_min=40,
        fats_max=80,
        fiber_min=20,
        fiber_max=40,
        sodium_min=0,
        sodium_max=2300,
        calorie_distribution_tolerance=0.07,
        min_distinct_categories=3,
        enable_optional_checks=True,
    )
