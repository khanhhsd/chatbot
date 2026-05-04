"""
Test script for the nutrition calculator module.
"""
from core.nutrition_calculator import calculate_nutrition_portions

def test_nutrition_calculator():
    """Test the nutrition calculator with sample data."""
    user_prompt = "i need recommendation for lunch, i need 42 grams of protein, 80 grams of carbs and some vegetables"
    rag_response = "For lunch, try chicken breast, rice, and broccoli. These foods provide a good balance of protein and carbohydrates."

    result = calculate_nutrition_portions(user_prompt, rag_response)

    print("=== Nutrition Calculator Test ===")
    print(f"User prompt: {user_prompt}")
    print(f"RAG response: {rag_response}")
    print(f"\nTarget needs: {result['target_needs']}")
    print(f"Foods found: {result['foods_found']}/{result['foods_requested']}")
    print("\nRecommended portions:")
    for food, portion in result['portions'].items():
        if portion > 0:
            print(f"  - {food}: {portion:.1f}g")

    print("\nActual nutrition:")
    actual = result['actual_nutrition']
    target = result['target_needs']
    for nutrient in ['protein', 'carbs', 'fat', 'calories']:
        if nutrient in target:
            print(f"  - {nutrient}: {actual.get(nutrient, 0):.1f}g (target: {target[nutrient]:.1f}g)")

if __name__ == "__main__":
    test_nutrition_calculator()