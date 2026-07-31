"""
Content-based food recommendation engine.

Builds a TF-IDF representation of each food item using its cuisine,
category, diet type and ingredients, then uses cosine similarity to
find the most similar dishes to a given item (or to a user's stated
preferences).
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class FoodRecommender:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self._build_model()

    def _build_model(self):
        # Combine descriptive text fields into a single "tag soup" per food
        self.df["tags"] = (
            (self.df["cuisine"] + " ") * 3
            + (self.df["category"] + " ") * 2
            + (self.df["diet"] + " ") * 2
            + self.df["ingredients"].str.replace(",", " ")
        ).str.lower()

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["tags"])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def recommend_similar(self, food_id: int, top_n: int = 5) -> pd.DataFrame:
        """Return top_n items most similar to the given food_id."""
        if food_id not in self.df["id"].values:
            return pd.DataFrame()

        idx = self.df.index[self.df["id"] == food_id][0]
        scores = list(enumerate(self.similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx][:top_n]

        result_idx = [i for i, _ in scores]
        result = self.df.iloc[result_idx].copy()
        result["similarity"] = [round(s * 100, 1) for _, s in scores]
        return result

    def recommend_by_preferences(
        self,
        cuisines=None,
        diet=None,
        max_calories=None,
        min_rating=0,
        max_price=None,
        top_n: int = 8,
    ) -> pd.DataFrame:
        """Filter + rank foods based on explicit user preferences."""
        result = self.df.copy()

        if cuisines:
            result = result[result["cuisine"].isin(cuisines)]
        if diet and diet != "Any":
            result = result[result["diet"] == diet]
        if max_calories:
            result = result[result["calories"] <= max_calories]
        if max_price:
            result = result[result["price"] <= max_price]
        result = result[result["rating"] >= min_rating]

        result = result.sort_values(by="rating", ascending=False)
        return result.head(top_n)

    def get_by_id(self, food_id: int):
        row = self.df[self.df["id"] == food_id]
        return row.iloc[0] if not row.empty else None
