from recommendation import recommend_food

foods = recommend_food(
    disease="Diabetes",
    meal_type="Breakfast",
    top_n=5
)

print(foods)