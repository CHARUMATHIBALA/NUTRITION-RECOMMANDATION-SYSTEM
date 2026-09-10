

from __future__ import annotations

import logging
import hashlib
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import pandas as pd

from models import food_df
import config
from backend.services.enhanced_recommender import EnhancedNutritionRecommender
from backend.services.severity_rules import (
    get_severity_explanation,
    get_recommendation_reason,
    _normalise_disease_key,
    _normalise_severity,
)

logger = logging.getLogger(__name__)


@dataclass
class FoodUsageTracker:
    """
    Tracks food usage across the 7-day meal plan for diversity control.

    Food IDs are the primary identity. Dish names are kept so existing
    tests and diversity metrics continue to work.
    """
    food_usage_count: Dict[str, int] = field(default_factory=dict)
    food_id_usage_count: Dict[int, int] = field(default_factory=dict)
    food_last_used_day: Dict[str, int] = field(default_factory=dict)
    food_meal_types: Dict[str, Set[str]] = field(default_factory=dict)
    used_meal_combinations: Dict[str, Set[Tuple[int, ...]]] = field(
        default_factory=lambda: defaultdict(set)
    )
    used_day_signatures: Set[Tuple[int, ...]] = field(default_factory=set)

    def record_usage(
        self,
        food_name: str,
        day: int,
        meal_type: str,
        food_id: Optional[int] = None,
    ):
        """Record that a food was used on a specific day and meal type."""
        self.food_usage_count[food_name] = self.food_usage_count.get(food_name, 0) + 1
        self.food_last_used_day[food_name] = day
        if food_name not in self.food_meal_types:
            self.food_meal_types[food_name] = set()
        self.food_meal_types[food_name].add(meal_type)
        if food_id is not None and int(food_id) != 0:
            fid = int(food_id)
            self.food_id_usage_count[fid] = self.food_id_usage_count.get(fid, 0) + 1

    def get_usage_count(self, food_name: str) -> int:
        """Get how many times a food has been used."""
        return self.food_usage_count.get(food_name, 0)

    def get_id_usage_count(self, food_id: int) -> int:
        """Get how many times a food_id has been used across the week."""
        return self.food_id_usage_count.get(int(food_id), 0)

    def get_days_since_last_use(self, food_name: str, current_day: int) -> int:
        """Get days since last use (0 if never used or used today)."""
        if food_name not in self.food_last_used_day:
            return 0  # Never used
        return current_day - self.food_last_used_day[food_name]

    def was_used_in_meal_type(self, food_name: str, meal_type: str) -> bool:
        """Check if food was used in a specific meal type."""
        return meal_type in self.food_meal_types.get(food_name, set())

    def record_meal_combination(self, meal_type: str, food_ids: Tuple[int, ...]):
        self.used_meal_combinations[meal_type.lower()].add(tuple(sorted(food_ids)))

    def combo_already_used(self, meal_type: str, food_ids: Tuple[int, ...]) -> bool:
        return tuple(sorted(food_ids)) in self.used_meal_combinations[meal_type.lower()]


@dataclass 
class WeeklyPlanResult:
    """Structured result for 7-day meal plan."""
    weekly_plan: List[Dict[str, List[Dict]]]
    diversity_metrics: Dict[str, any]
    safety_report: Dict[str, any]
    nutrition_summary: Dict[str, any]
    scoring_info: Dict[str, any]
    meal_validation: Dict[str, any] = field(default_factory=dict)


class WeeklyMealPlanner:
    """
    7-day diverse meal planner with repetition control and category diversity.
    
    Pipeline:
    USER PROFILE → DISEASE/SEVERITY FILTERS → NUTRITION/CONTENT/HYBRID SCORING 
    → DIVERSITY ENGINE → 7-DAY PLANNING → DAILY NUTRITION BALANCE → OUTPUT
    """
    
    MEAL_TYPES = ['Breakfast', 'Lunch', 'Snack', 'Dinner']
    DAYS = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7']
    
    # Meal calorie distribution
    MEAL_CALORIE_DISTRIBUTION = {
        'Breakfast': 0.25,
        'Lunch': 0.35,
        'Snack': 0.15,
        'Dinner': 0.25,
    }
    
    # Diversity penalty configuration
    CONSECUTIVE_DAY_PENALTY = 0.6      # Strong penalty for same food on consecutive days
    SAME_DAY_PENALTY = 0.8              # Very strong penalty for same food twice in one day
    REPEAT_PENALTY = 0.2                # Base penalty for each additional use
    DIVERSITY_BONUS = 0.1              # Bonus for food category diversity
    MAX_FOOD_USES = 1                   # Enforce unique foods across the week

    def __init__(self):
        self.base_recommender = EnhancedNutritionRecommender()
        self.usage_tracker = FoodUsageTracker()
        self._condiment_fiber_cutoff: Optional[float] = None
        
    def _get_food_category(self, food_name: str) -> str:
        """
        Extract food category from dish name for diversity considerations.
        Categories based on common South Indian food patterns.
        """
        food_lower = food_name.lower()
        
        # Rice-based dishes
        if any(keyword in food_lower for keyword in ['rice', 'biriyani', 'pulao', 'khichdi']):
            return 'rice_based'
        
        # Wheat/bread-based
        if any(keyword in food_lower for keyword in ['roti', 'chapati', 'paratha', 'bread', 'sandwich']):
            return 'wheat_based'
        
        # Millet-based
        if any(keyword in food_lower for keyword in ['daliya', 'ragi', 'millet']):
            return 'millet_based'
        
        # Protein-rich
        if any(keyword in food_lower for keyword in ['dal', 'paneer', 'chicken', 'egg', 'fish', 'keema']):
            return 'protein_rich'
        
        # Vegetable-based
        if any(keyword in food_lower for keyword in ['vegetable', 'veg', 'sabzi', 'curry']):
            return 'vegetable_based'
        
        # Breakfast-specific
        if any(keyword in food_lower for keyword in ['idli', 'dosa', 'upma', 'pongal', 'porridge']):
            return 'breakfast_dish'
        
        # Snack/beverage
        if any(keyword in food_lower for keyword in ['shake', 'juice', 'tea', 'coffee', 'lassi', 'drink']):
            return 'beverage_snack'
        
        # Default
        return 'other'
    
    def _calculate_diversity_score(
        self, 
        food_name: str, 
        day: int, 
        meal_type: str,
        current_day_categories: Set[str]
    ) -> float:
        """
        Calculate diversity score for a food item.
        
        Returns penalty (0.0 to 1.0) where higher means more penalty (less diverse).
        """
        penalty = 0.0
        
        # Check consecutive day usage
        days_since_use = self.usage_tracker.get_days_since_last_use(food_name, day)
        if food_name not in self.usage_tracker.food_last_used_day:
            # Never used before - no penalty
            pass
        elif days_since_use == 0:
            # Used today - very strong penalty
            penalty += self.SAME_DAY_PENALTY
        elif days_since_use == 1:
            # Used yesterday - strong penalty
            penalty += self.CONSECUTIVE_DAY_PENALTY
        
        # Check overall repetition
        usage_count = self.usage_tracker.get_usage_count(food_name)
        if usage_count > 0:
            penalty += (usage_count - 1) * self.REPEAT_PENALTY
        
        # Category diversity bonus
        food_category = self._get_food_category(food_name)
        if food_category not in current_day_categories:
            # Bonus for introducing new category
            penalty -= self.DIVERSITY_BONUS
        
        return max(0.0, min(1.0, penalty))

    def _get_condiment_fiber_cutoff(self) -> float:
        """
        Dataset-driven cutoff for condiment/spice-only Snack items.

        Dry spice and seasoning powders in this dataset are labelled MealType=Snack
        but sit as fibre outliers (mean + 3 std of exact-Snack rows). This uses
        MealType + nutrition columns only — no dish-name allow/deny list.
        """
        if self._condiment_fiber_cutoff is not None:
            return self._condiment_fiber_cutoff
        df = self.base_recommender.df
        if df is None or df.empty or 'MealType' not in df.columns:
            self._condiment_fiber_cutoff = 12.0
            return self._condiment_fiber_cutoff
        meal_l = df['MealType'].astype(str).str.strip().str.lower()
        fiber = pd.to_numeric(df.get('Fibre (g)', 0), errors='coerce').fillna(0.0)
        snack_fiber = fiber[meal_l == 'snack']
        if len(snack_fiber) >= 10 and float(snack_fiber.std() or 0) > 0:
            cutoff = float(snack_fiber.mean() + 3.0 * snack_fiber.std())
        else:
            cutoff = 12.0
        self._condiment_fiber_cutoff = cutoff
        return cutoff

    def _exclude_condiment_only_items(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove spice/seasoning powders that cannot be a standalone meal."""
        if df.empty or 'MealType' not in df.columns:
            return df
        meal_l = df['MealType'].astype(str).str.strip().str.lower()
        fiber = pd.to_numeric(df.get('Fibre (g)', 0), errors='coerce').fillna(0.0)
        cutoff = self._get_condiment_fiber_cutoff()
        condiment_mask = meal_l.eq('snack') & (fiber >= cutoff)
        filtered = df.loc[~condiment_mask].copy()
        # If the filter would empty the slot, keep the original safe pool.
        return filtered if not filtered.empty else df.copy()

    @staticmethod
    def _food_id_key(row: pd.Series) -> int:
        fid = row.get('food_id', 0)
        try:
            fid_int = int(fid)
        except (TypeError, ValueError):
            fid_int = 0
        if fid_int:
            return fid_int
        name = str(row.get('Dish Name', ''))
        digest = hashlib.md5(name.encode('utf-8')).hexdigest()
        return int(digest[:8], 16)

    def _filter_by_meal_type_deterministic(
        self, 
        df: pd.DataFrame, 
        meal_type: str
    ) -> pd.DataFrame:
        """
        Filter foods by meal type with deterministic handling of combined types.
        Then drop condiment-only items that are not standalone meals.
        """
        meal_lower = meal_type.lower()
        
        # Handle combined meal types like "Lunch/Dinner"
        if meal_type == 'Lunch':
            # Include both Lunch and Lunch/Dinner
            mask = df["MealType"].str.lower().str.contains(
                r'^lunch$|^lunch/dinner$', na=False, regex=True
            )
        elif meal_type == 'Dinner':
            # Include both Dinner and Lunch/Dinner
            mask = df["MealType"].str.lower().str.contains(
                r'^dinner$|^lunch/dinner$', na=False, regex=True
            )
        elif meal_type == 'Breakfast':
            mask = df["MealType"].str.lower().str.contains(
                r'^breakfast$', na=False, regex=True
            )
        elif meal_type == 'Snack':
            mask = df["MealType"].str.lower().str.contains(
                r'^snack$', na=False, regex=True
            )
        else:
            mask = df["MealType"].str.lower().str.contains(meal_lower, na=False)

        meal_df = df[mask].copy()
        return self._exclude_condiment_only_items(meal_df)

    def _candidate_passes(
        self,
        row: pd.Series,
        used_today: Set[int],
        allow_today_repeat: bool,
        allow_over_cap: bool,
    ) -> bool:
        fid = self._food_id_key(row)
        usage = self.usage_tracker.get_id_usage_count(fid)
        if not allow_over_cap and usage >= self.MAX_FOOD_USES:
            return False
        if not allow_today_repeat and fid in used_today:
            return False
        return True

    def _iter_meal_combos(
        self,
        ranked: List[pd.Series],
        num_foods: int,
        used_today: Set[int],
        allow_today_repeat: bool,
        allow_over_cap: bool,
        allow_combo_repeat: bool,
        meal_type: str,
    ) -> Optional[List[pd.Series]]:
        """Deterministic first feasible combination in ranked order."""
        eligible = [
            row for row in ranked
            if self._candidate_passes(row, used_today, allow_today_repeat, allow_over_cap)
        ]
        if len(eligible) < num_foods:
            return None

        n = len(eligible)
        if num_foods == 1:
            for row in eligible:
                fid = self._food_id_key(row)
                combo = (fid,)
                if allow_combo_repeat or not self.usage_tracker.combo_already_used(meal_type, combo):
                    return [row]
            return None

        # num_foods >= 2: combinations in ranked order (i < j keeps higher-ranked first)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = eligible[i], eligible[j]
                ids = (self._food_id_key(a), self._food_id_key(b))
                if ids[0] == ids[1]:
                    continue
                if allow_combo_repeat or not self.usage_tracker.combo_already_used(meal_type, ids):
                    return [a, b]
        return None

    def _select_foods_for_meal_slot(
        self,
        meal_df: pd.DataFrame,
        day: int,
        meal_type: str,
        user_profile: Dict,
        num_foods: int,
        current_day_categories: Set[str],
        current_day_food_ids: Set[int],
    ) -> List[pd.Series]:
        """
        Select foods for a specific meal slot with diversity as a soft preference.

        Ranking is deterministic: unused food_ids first, then hybrid score,
        then food_id. No random selection. Safety pool is already disease/severity
        filtered by the caller; this method never adds unsafe foods.
        """
        if meal_df.empty:
            return []

        meal_profile = {
            **user_profile,
            '_meal_calorie_fraction': self.MEAL_CALORIE_DISTRIBUTION.get(meal_type, 0.25),
        }

        hybrid_scores, breakdowns = [], []
        for _, row in meal_df.iterrows():
            h, bd = self.base_recommender._calculate_hybrid_score(row, meal_profile)
            hybrid_scores.append(h)
            breakdowns.append(bd)

        meal_df = meal_df.copy()
        meal_df['hybrid_score'] = hybrid_scores
        meal_df['_score_breakdown'] = breakdowns
        meal_df['food_id_key'] = meal_df.apply(self._food_id_key, axis=1)
        meal_df['usage'] = meal_df['food_id_key'].map(
            self.usage_tracker.get_id_usage_count
        )
        meal_df['diversity_penalty'] = meal_df['Dish Name'].apply(
            lambda name: self._calculate_diversity_score(
                name, day, meal_type, current_day_categories
            )
        )
        meal_df['final_score'] = meal_df['hybrid_score'] - meal_df['diversity_penalty']

        # Unused foods always rank above used foods (preference, not a safety rule).
        meal_df = meal_df.sort_values(
            ['usage', 'final_score', 'food_id_key'],
            ascending=[True, False, True],
            kind='mergesort',
        )
        ranked = [row for _, row in meal_df.iterrows()]

        selected = None
        for allow_today, allow_over, allow_combo in (
            (False, False, False),
            (True, False, False),
            (False, True, False),
            (True, True, False),
            (True, True, True),
        ):
            selected = self._iter_meal_combos(
                ranked, num_foods, current_day_food_ids,
                allow_today, allow_over, allow_combo, meal_type,
            )
            if selected:
                break

        if not selected:
            selected = ranked[:num_foods]

        result = []
        combo_ids: List[int] = []
        for row in selected:
            food_name = str(row.get('Dish Name', ''))
            fid = self._food_id_key(row)
            combo_ids.append(fid)
            self.usage_tracker.record_usage(food_name, day, meal_type, food_id=fid)
            current_day_food_ids.add(fid)
            current_day_categories.add(self._get_food_category(food_name))

            bd = dict(row.get('_score_breakdown', {}) or {})
            bd['diversity_penalty'] = round(float(row.get('diversity_penalty', 0.0)), 4)
            bd['final_score'] = round(float(row.get('final_score', 0.0)), 4)
            row = row.copy()
            row['_score_breakdown'] = bd
            result.append(row)

        if combo_ids:
            self.usage_tracker.record_meal_combination(meal_type, tuple(combo_ids))
        return result
    
    def _calculate_daily_nutrition_totals(
        self, 
        day_plan: Dict[str, List[pd.Series]]
    ) -> Dict[str, float]:
        """Calculate total nutrition for a day."""
        totals = {
            'calories': 0.0,
            'protein': 0.0,
            'carbohydrates': 0.0,
            'fats': 0.0,
            'fiber': 0.0,
            'sodium': 0.0,
        }
        
        for foods in day_plan.values():
            for food in foods:
                totals['calories'] += float(food.get('Calories (kcal)', 0))
                totals['protein'] += float(food.get('Protein (g)', 0))
                totals['carbohydrates'] += float(food.get('Carbohydrates (g)', 0))
                totals['fats'] += float(food.get('Fats (g)', 0))
                totals['fiber'] += float(food.get('Fibre (g)', 0))
                totals['sodium'] += float(food.get('Sodium (mg)', 0))
        
        return {k: round(v, 2) for k, v in totals.items()}
    
    def _check_safety_filters(
        self,
        filtered_df: pd.DataFrame,
        original_df: pd.DataFrame,
        diseases: List[str],
        severity_dict: Dict
    ) -> Dict[str, any]:
        """
        Generate safety report for filtered foods.
        """
        return {
            'original_food_count': len(original_df),
            'filtered_food_count': len(filtered_df),
            'foods_removed': len(original_df) - len(filtered_df),
            'diseases_filtered': diseases,
            'severity_levels': severity_dict,
            'filtering_applied': True
        }
    
    def _calculate_diversity_metrics(
        self, 
        weekly_plan: List[Dict[str, List[Dict]]]
    ) -> Dict[str, any]:
        """
        Calculate diversity metrics for the weekly plan.
        """
        all_foods = []
        food_counts = defaultdict(int)
        
        for day_plan in weekly_plan:
            # Handle both dict formats (with 'meals' key and direct meal dict)
            if 'meals' in day_plan:
                meals_dict = day_plan['meals']
            else:
                meals_dict = day_plan
                
            for meal_type, foods in meals_dict.items():
                for food in foods:
                    # Handle both dict and Series formats
                    if isinstance(food, dict):
                        food_name = food.get('Dish Name', '')
                    elif hasattr(food, 'get'):
                        food_name = food.get('Dish Name', '')
                    else:
                        continue
                    
                    if food_name:
                        all_foods.append(food_name)
                        food_counts[food_name] += 1
        
        total_slots = len(all_foods)
        unique_foods = len(set(all_foods))
        
        # Calculate repetition statistics
        repeated_foods = [f for f, c in food_counts.items() if c > 1]
        max_repetition = max(food_counts.values()) if food_counts else 0
        
        return {
            'total_meal_slots': total_slots,
            'unique_foods': unique_foods,
            'unique_food_percentage': round((unique_foods / total_slots * 100) if total_slots > 0 else 0, 2),
            'repeated_foods_count': len(repeated_foods),
            'repeated_foods': repeated_foods[:10],  # First 10 for brevity
            'max_repetition_count': max_repetition,
            'average_repetition': round(sum(food_counts.values()) / len(food_counts), 2) if food_counts else 0
        }

    def validate_weekly_plan(
        self,
        weekly_plan: List[Dict],
        diversity_metrics: Dict,
        filtered_df: pd.DataFrame,
    ) -> Dict:
        """Post-generation checks for the 7-day plan. Does not change safety rules."""
        errors: List[str] = []
        warnings: List[str] = []
        checks: Dict[str, bool] = {}

        checks['exactly_7_days'] = len(weekly_plan) == 7
        if not checks['exactly_7_days']:
            errors.append(f"Expected 7 days, got {len(weekly_plan)}")

        required = {'breakfast', 'lunch', 'snack', 'dinner'}
        missing_slots = []
        meal_type_ok = True
        condiment_ok = True
        cutoff = self._get_condiment_fiber_cutoff()
        day_signatures: List[Tuple] = []
        meal_combos: Dict[str, List[Tuple]] = defaultdict(list)
        food_counts: Dict[str, int] = defaultdict(int)

        for day_plan in weekly_plan:
            meals = day_plan.get('meals', {})
            absent = required - set(meals.keys())
            if absent:
                missing_slots.append((day_plan.get('day'), sorted(absent)))
            day_foods: List[str] = []
            for meal_type, foods in meals.items():
                if not foods:
                    missing_slots.append((day_plan.get('day'), [meal_type]))
                combo = tuple(f.get('Dish Name', '') for f in foods)
                meal_combos[meal_type].append(combo)
                for food in foods:
                    name = food.get('Dish Name', '')
                    day_foods.append(name)
                    food_counts[name] += 1
                    mt = str(food.get('MealType', '')).lower()
                    slot = meal_type.lower()
                    compatible = (
                        slot in mt
                        or (slot in ('lunch', 'dinner') and 'lunch/dinner' in mt)
                    )
                    if foods and not compatible:
                        meal_type_ok = False
                    fiber = float(food.get('Fibre (g)', 0) or 0)
                    if mt.strip() == 'snack' and fiber >= cutoff:
                        condiment_ok = False
            day_signatures.append(tuple(day_foods))

        checks['all_meal_slots_present'] = not missing_slots
        if missing_slots:
            errors.append(f"Missing meal slots: {missing_slots[:5]}")

        checks['meal_type_suitability'] = meal_type_ok
        if not meal_type_ok:
            errors.append("One or more foods do not match their meal-type slot")

        checks['no_condiment_standalone_meals'] = condiment_ok
        if not condiment_ok:
            errors.append("Condiment/spice-only item selected as a standalone meal")

        duplicate_days = len(day_signatures) != len(set(day_signatures))
        checks['no_duplicate_day_combinations'] = not duplicate_days
        if duplicate_days:
            errors.append("Duplicate complete-day meal combinations found")

        duplicate_meals = False
        for slot, combos in meal_combos.items():
            nonempty = [c for c in combos if c]
            if len(nonempty) != len(set(nonempty)):
                duplicate_meals = True
        checks['no_duplicate_meal_combinations'] = not duplicate_meals
        if duplicate_meals:
            warnings.append(
                "The same meal combination appears on more than one day "
                "(allowed only when the safe candidate pool is too small)"
            )

        max_rep = diversity_metrics.get('max_repetition_count', 0)
        over_cap = [name for name, c in food_counts.items() if c > self.MAX_FOOD_USES]
        safe_pool_size = len(filtered_df) if filtered_df is not None else 0
        checks['no_excessive_food_repetition'] = (
            max_rep <= self.MAX_FOOD_USES or safe_pool_size < 42
        )
        if over_cap and safe_pool_size >= 42:
            warnings.append(
                f"Foods used more than {self.MAX_FOOD_USES} times: {over_cap[:8]}"
            )
        elif over_cap:
            warnings.append(
                f"Controlled repetition used because the safe pool has "
                f"{safe_pool_size} foods: {over_cap[:8]}"
            )

        checks['deterministic_selection'] = True

        return {
            'is_valid': len(errors) == 0,
            'warnings': warnings,
            'errors': errors,
            'checks': checks,
        }
    
    def generate_weekly_plan(
        self,
        user_profile: Dict
    ) -> WeeklyPlanResult:
        """
        Generate a complete 7-day diverse meal plan.
        
        Args:
            user_profile: User profile including diseases, severity, BMI, etc.
            
        Returns:
            WeeklyPlanResult with structured 7-day plan and metrics
        """
        # Reset usage tracker for new plan
        self.usage_tracker = FoodUsageTracker()
        
        # Extract user profile data
        diseases = user_profile.get('diseases', [])
        if isinstance(diseases, str):
            diseases = [diseases]
        
        severity_dict = user_profile.get('severity', {})
        daily_calories = user_profile.get('daily_calories', 2000)
        
        # Apply safety filters (preserve existing pipeline)
        filtered_df = self.base_recommender._apply_disease_filters(
            self.base_recommender.df, diseases
        )
        
        if not filtered_df.empty:
            filtered_df = self.base_recommender._apply_severity_filters(
                filtered_df, diseases, severity_dict
            )
        
        # Safety report
        safety_report = self._check_safety_filters(
            filtered_df, self.base_recommender.df, diseases, severity_dict
        )
        
        # Generate weekly plan
        weekly_plan = []
        weekly_nutrition_summary = []
        
        for day_idx, day_name in enumerate(self.DAYS, start=1):
            day_plan = {}
            current_day_categories = set()
            current_day_food_ids: Set[int] = set()
            
            for meal_type in self.MEAL_TYPES:
                # Filter by meal type
                meal_df = self._filter_by_meal_type_deterministic(filtered_df, meal_type)
                
                # Determine number of foods per meal
                num_foods = 2 if meal_type in ['Lunch', 'Dinner'] else 1
                
                # Select foods with diversity consideration
                selected_foods = self._select_foods_for_meal_slot(
                    meal_df, day_idx, meal_type, user_profile, 
                    num_foods, current_day_categories, current_day_food_ids
                )
                
                # Convert to dict format for JSON serialization
                meal_foods = []
                for food in selected_foods:
                    food_dict = {
                        'food_id': int(float(food.get('food_id', 0) or 0)),
                        'Dish Name': str(food.get('Dish Name', '')),
                        'Calories (kcal)': float(food.get('Calories (kcal)', 0)),
                        'Protein (g)': float(food.get('Protein (g)', 0)),
                        'Carbohydrates (g)': float(food.get('Carbohydrates (g)', 0)),
                        'Fats (g)': float(food.get('Fats (g)', 0)),
                        'Fibre (g)': float(food.get('Fibre (g)', 0)),
                        'Sodium (mg)': float(food.get('Sodium (mg)', 0)),
                        'MealType': str(food.get('MealType', '')),
                        '_score_breakdown': food.get('_score_breakdown', {})
                    }
                    meal_foods.append(food_dict)
                
                day_plan[meal_type.lower()] = meal_foods
            
            # Calculate daily nutrition
            # Convert dict foods to Series for nutrition calculation
            series_day_plan = {}
            for meal_type, foods in day_plan.items():
                if foods and isinstance(foods, list):
                    series_day_plan[meal_type] = [pd.Series(food) for food in foods]
                else:
                    series_day_plan[meal_type] = []
            
            daily_nutrition = self._calculate_daily_nutrition_totals(series_day_plan)
            daily_nutrition['day'] = day_name
            weekly_nutrition_summary.append(daily_nutrition)
            
            weekly_plan.append({
                'day': day_name,
                'day_number': day_idx,
                'meals': day_plan,
                'daily_nutrition': daily_nutrition
            })
            day_ids = tuple(sorted(current_day_food_ids))
            self.usage_tracker.used_day_signatures.add(day_ids)
        
        # Calculate metrics
        diversity_metrics = self._calculate_diversity_metrics(weekly_plan)
        
        # Overall nutrition summary
        overall_nutrition = {
            'average_daily_calories': round(
                sum(d['calories'] for d in weekly_nutrition_summary) / 7, 2
            ),
            'average_daily_protein': round(
                sum(d['protein'] for d in weekly_nutrition_summary) / 7, 2
            ),
            'average_daily_carbs': round(
                sum(d['carbohydrates'] for d in weekly_nutrition_summary) / 7, 2
            ),
            'average_daily_fats': round(
                sum(d['fats'] for d in weekly_nutrition_summary) / 7, 2
            ),
            'average_daily_fiber': round(
                sum(d['fiber'] for d in weekly_nutrition_summary) / 7, 2
            ),
            'target_daily_calories': daily_calories
        }
        
        # Scoring info
        scoring_info = {
            'ncf_available': self.base_recommender._ncf_available,
            'w_nutrition': round(self.base_recommender._eff_w_nutrition, 4),
            'w_content': round(self.base_recommender._eff_w_content, 4),
            'w_ncf': round(self.base_recommender._eff_w_ncf, 4),
            'diversity_penalties_applied': True,
            'deterministic_selection': True
        }
        
        meal_validation = self.validate_weekly_plan(
            weekly_plan, diversity_metrics, filtered_df
        )

        return WeeklyPlanResult(
            weekly_plan=weekly_plan,
            diversity_metrics=diversity_metrics,
            safety_report=safety_report,
            nutrition_summary=overall_nutrition,
            scoring_info=scoring_info,
            meal_validation=meal_validation,
        )


def generate_weekly_recommendations(user_profile: Dict) -> Dict:
    """
    Convenience function to generate 7-day meal recommendations.
    
    Args:
        user_profile: Complete user profile with health data
        
    Returns:
        Dictionary with structured 7-day plan and metrics
    """
    planner = WeeklyMealPlanner()
    result = planner.generate_weekly_plan(user_profile)
    
    return {
        'weekly_plan': result.weekly_plan,
        'diversity_metrics': result.diversity_metrics,
        'safety_report': result.safety_report,
        'nutrition_summary': result.nutrition_summary,
        'scoring_info': result.scoring_info,
        'meal_validation': result.meal_validation,
    }


if __name__ == "__main__":
    # Test the weekly planner
    test_profile = {
        'age': 45,
        'gender': 'Male',
        'weight': 75,
        'height': 175,
        'bmi': 24.5,
        'activity_level': 'Moderate',
        'diseases': ['Diabetes'],
        'severity': {'diabetes': 'moderate'},
        'daily_calories': 2000
    }
    
    result = generate_weekly_recommendations(test_profile)
    print("7-Day Meal Plan Generated Successfully!")
    print(f"Unique Foods: {result['diversity_metrics']['unique_foods']}")
    print(f"Unique Food Percentage: {result['diversity_metrics']['unique_food_percentage']}%")