"""
Nutrition Calculator Module

This module provides functionality to calculate food portions based on nutritional needs.
It parses user requirements from prompts and optimizes food quantities from RAG recommendations.
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import os


class NutritionCalculator:
    """
    Calculator for determining optimal food portions to meet nutritional goals.
    """

    def __init__(self, db_path: str = None):
        """
        Initialize the calculator with food database.

        Args:
            db_path: Path to the food database directory
        """
        self.db_path = db_path or r"D:\hocj\AI\TTCS\DataBase\archive\FINAL FOOD DATASET"
        self.food_data = self._load_food_database()

    def _load_food_database(self) -> pd.DataFrame:
        """Load and combine all food data CSV files."""
        csv_files = [
            "FOOD-DATA-GROUP1.csv",
            "FOOD-DATA-GROUP2.csv",
            "FOOD-DATA-GROUP3.csv",
            "FOOD-DATA-GROUP4.csv",
            "FOOD-DATA-GROUP5.csv"
        ]

        all_data = []
        for csv_file in csv_files:
            filepath = os.path.join(self.db_path, csv_file)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                all_data.append(df)

        if not all_data:
            raise FileNotFoundError(f"No food data files found in {self.db_path}")

        combined_df = pd.concat(all_data, ignore_index=True)

        # Clean column names
        combined_df.columns = combined_df.columns.str.strip()

        # Ensure 'food' column exists and is lowercase for matching
        if 'food' in combined_df.columns:
            combined_df['food'] = combined_df['food'].str.lower().str.strip()

        return combined_df

    def parse_nutrition_needs(self, text: str) -> Dict[str, float]:
        """
        Parse nutritional requirements from user text.

        Args:
            text: User input text containing nutrition needs

        Returns:
            Dict mapping nutrient names to target amounts (in grams)
        """
        needs = {}

        # Patterns for different nutrients - more flexible matching
        patterns = {
            'protein': r'(\d+(?:\.\d+)?)\s*g(?:rams?)?\s*(?:of\s*)?protein',
            'carbs': r'(\d+(?:\.\d+)?)\s*g(?:rams?)?\s*(?:of\s*)?(?:carbs?|carbohydrates?)',
            'fat': r'(\d+(?:\.\d+)?)\s*g(?:rams?)?\s*(?:of\s*)?fat',
            'calories': r'(\d+(?:\.\d+)?)\s*calories?',
            'fiber': r'(\d+(?:\.\d+)?)\s*g(?:rams?)?\s*(?:of\s*)?(?:fiber|fibre)',
            'sugar': r'(\d+(?:\.\d+)?)\s*g(?:rams?)?\s*(?:of\s*)?sugar',
        }

        text_lower = text.lower()

        for nutrient, pattern in patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                needs[nutrient] = float(match.group(1))

        return needs

    def parse_meal_foods(self, text: str) -> List[str]:
        """
        Parse food items from RAG response or meal description.

        Args:
            text: Text containing food items

        Returns:
            List of food names
        """
        if not text:
            return []
            
        # Remove common phrases and numbers with units
        text = re.sub(r'(?:for lunch|for dinner|try|eat|have|recommend|meal|includes?)', '', text, flags=re.IGNORECASE)
        
        # Split by common separators
        foods = re.split(r'[,;•\n]|\s+and\s+|\s+or\s+|\)', text)

        # Clean and filter
        cleaned_foods = []
        for food in foods:
            # Remove parentheses and their contents
            food = re.sub(r'\([^)]*\)', '', food)
            food = food.strip()
            
            # Remove quantities and units (e.g., "10g", "500 grams", "2 cups")
            food = re.sub(r'\d+(?:\.\d+)?\s*(?:g|grams?|kg|kilograms?|oz|ounces?|cups?|lbs?|pounds?|ml|l|liters?)', '', food, flags=re.IGNORECASE)
            
            # Remove remaining numbers and special characters at boundaries
            food = re.sub(r'^\d+\s*', '', food)  # Remove leading numbers
            food = re.sub(r'^\W+|\W+$', '', food)  # Remove special chars at boundaries
            
            food = food.strip().lower()

            # Only keep meaningful food names (not too short, not empty)
            if food and len(food) > 2 and not re.match(r'^[\d\s]+$', food):
                cleaned_foods.append(food)

        return list(set(cleaned_foods))  # Remove duplicates

    def get_food_nutrition(self, food_name: str) -> Optional[Dict[str, float]]:
        """
        Get nutritional data for a food item.

        Args:
            food_name: Name of the food

        Returns:
            Dict with nutritional values per 100g, or None if not found
        """
        food_name = food_name.lower().strip()

        # Try exact match first
        exact_matches = self.food_data[self.food_data['food'].str.lower() == food_name]
        if not exact_matches.empty:
            row = exact_matches.iloc[0]
        else:
            # Try partial matching with multiple strategies
            candidates = []

            # Strategy 1: Contains all major words
            words = [w for w in food_name.split() if len(w) > 2 and w not in ['with', 'and', 'the', 'for', 'from']]
            if words:
                matches = self.food_data.copy()
                for word in words:
                    matches = matches[matches['food'].str.contains(re.escape(word), case=False, na=False, regex=True)]
                if not matches.empty:
                    candidates.extend(matches.to_dict('records'))

            # Strategy 2: Find foods with similar names (simple fuzzy matching)
            if not candidates:
                # Look for foods that share at least 2 words
                food_words = set(food_name.split())
                for _, row in self.food_data.iterrows():
                    db_food = row['food'].lower()
                    db_words = set(db_food.split())
                    common_words = food_words.intersection(db_words)
                    if len(common_words) >= 2:  # At least 2 words in common
                        candidates.append(row.to_dict())

            if not candidates:
                return None

            # Choose the best candidate (prefer higher protein/carbs for nutrition calculation)
            best_candidate = None
            best_score = -1

            for candidate in candidates:
                score = 0
                if pd.notna(candidate.get('Protein', 0)):
                    score += float(candidate.get('Protein', 0))
                if pd.notna(candidate.get('Carbohydrates', 0)):
                    score += float(candidate.get('Carbohydrates', 0))
                if score > best_score:
                    best_score = score
                    best_candidate = candidate

            if best_candidate:
                row = pd.Series(best_candidate)
            else:
                return None

        nutrition = {}
        nutrient_columns = {
            'protein': 'Protein',
            'carbs': 'Carbohydrates',
            'fat': 'Fat',
            'calories': 'Caloric Value',
            'fiber': 'Dietary Fiber',
            'sugar': 'Sugars'
        }

        for nutrient, col in nutrient_columns.items():
            if col in row.index and pd.notna(row[col]):
                nutrition[nutrient] = float(row[col])

        # Normalize extremely high calorie values that are likely stored per kg
        # instead of per 100g. If a food has more than 500 kcal per 100g, treat
        # the source numbers as per 1000g and convert them back to 100g values.
        if nutrition.get('calories', 0) > 500:
            for nutrient in list(nutrition.keys()):
                nutrition[nutrient] = nutrition[nutrient] / 10.0

        return nutrition

    def calculate_portions(self, foods: List[str], needs: Dict[str, float],
                          max_portion: float = 300.0) -> Dict[str, float]:
        """
        Calculate optimal portions for foods to meet nutritional needs.
        Uses a simple approach: assign the best food for each nutrient.

        Args:
            foods: List of food names
            needs: Dict of nutrient needs (grams)
            max_portion: Maximum portion per food (grams)

        Returns:
            Dict mapping food names to portion sizes (grams)
        """
        # Get nutrition data for each food
        food_nutrition = {}
        valid_foods = []

        for food in foods:
            nutrition = self.get_food_nutrition(food)
            if nutrition:
                food_nutrition[food] = nutrition
                valid_foods.append(food)

        if not valid_foods:
            return {}

        portions = {food: 0.0 for food in valid_foods}

        # For each nutrient, find the best food and calculate portion
        for nutrient, target in needs.items():
            if nutrient not in ['protein', 'carbs', 'fat', 'fiber', 'sugar']:
                continue

            # Find the food with the highest nutrient density for this nutrient
            best_food = None
            best_density = 0

            for food in valid_foods:
                if nutrient in food_nutrition[food] and food_nutrition[food][nutrient] > 0:
                    density = food_nutrition[food][nutrient]
                    if density > best_density:
                        best_density = density
                        best_food = food

            if best_food and best_density > 0:
                # Calculate portion needed, but cap at reasonable amount
                portion_needed = min((target / best_density) * 100, max_portion)
                portions[best_food] = max(portions[best_food], portion_needed)

        return portions

    def calculate_meal_nutrition(self, portions: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate total nutrition for a meal with given portions.

        Args:
            portions: Dict of food: portion (grams)

        Returns:
            Dict of total nutrients
        """
        totals = {'protein': 0, 'carbs': 0, 'fat': 0, 'calories': 0, 'fiber': 0, 'sugar': 0}

        for food, portion in portions.items():
            nutrition = self.get_food_nutrition(food)
            if nutrition:
                for nutrient in totals.keys():
                    if nutrient in nutrition:
                        totals[nutrient] += (nutrition[nutrient] * portion / 100)

        return totals

    def optimize_meal(self, user_prompt: str, rag_response: str) -> dict:
        """
        Main method: parse needs and foods, calculate optimal portions.

        Args:
            user_prompt: User's nutritional requirements
            rag_response: RAG system's meal recommendation

        Returns:
            Dict with portions and nutrition summary
        """
        # Parse nutritional needs
        needs = self.parse_nutrition_needs(user_prompt)

        # Parse foods from RAG response
        foods = self.parse_meal_foods(rag_response)

        # Calculate portions
        portions = self.calculate_portions(foods, needs)

        # Calculate actual nutrition
        actual_nutrition = self.calculate_meal_nutrition(portions)

        return {
            'portions': portions,
            'target_needs': needs,
            'actual_nutrition': actual_nutrition,
            'foods_found': len(portions),
            'foods_requested': len(foods)
        }


# Convenience function for external use
def calculate_nutrition_portions(user_prompt: str, rag_response: str) -> dict:
    """
    Calculate food portions based on user needs and RAG meal recommendation.

    Args:
        user_prompt: User's nutritional requirements
        rag_response: RAG system's meal recommendation

    Returns:
        Dict with portions and nutrition info
    """
    calculator = NutritionCalculator()
    return calculator.optimize_meal(user_prompt, rag_response)