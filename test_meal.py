from meal_planner import generate_meal_plan

meal_plan = generate_meal_plan("Diabetes")

for meal, foods in meal_plan.items():

    print("\n", "=" * 50)
    print(meal)
    print("=" * 50)

    print(foods[["Dish Name", "Calories (kcal)"]])