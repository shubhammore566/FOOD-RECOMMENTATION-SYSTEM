"""
Apriori-based "frequently bought together" recommendation engine.

Mirrors the market-basket analysis workflow from the Zomato Apriori
notebook: encode order transactions -> mine frequent itemsets ->
build association rules -> recommend items given something a
customer already ordered.
"""

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


class AprioriRecommender:
    def __init__(self, transactions: list, min_support: float = 0.20, min_confidence: float = 0.5):
        self.transactions = transactions
        self.min_support = min_support
        self.min_confidence = min_confidence
        self._build_model()

    def _build_model(self):
        te = TransactionEncoder()
        encoded = te.fit(self.transactions).transform(self.transactions)
        self.encoded_df = pd.DataFrame(encoded, columns=te.columns_)

        self.frequent_itemsets = apriori(
            self.encoded_df,
            min_support=self.min_support,
            use_colnames=True,
        ).sort_values(by="support", ascending=False).reset_index(drop=True)

        if len(self.frequent_itemsets) == 0:
            self.rules = pd.DataFrame(
                columns=["antecedents", "consequents", "support", "confidence", "lift"]
            )
        else:
            self.rules = association_rules(
                self.frequent_itemsets,
                metric="confidence",
                min_threshold=self.min_confidence,
            ).sort_values(by="confidence", ascending=False).reset_index(drop=True)

    @property
    def all_items(self):
        return sorted(self.encoded_df.columns.tolist())

    def recommend(self, item: str) -> pd.DataFrame:
        """Return items frequently ordered alongside `item`, ranked by confidence."""
        if self.rules.empty:
            return pd.DataFrame()

        rec = self.rules[self.rules["antecedents"].apply(lambda x: item in x)].copy()
        rec = rec.sort_values(by="confidence", ascending=False)
        rec["recommended_items"] = rec["consequents"].apply(lambda x: ", ".join(sorted(x)))
        return rec[["recommended_items", "support", "confidence", "lift"]]

    def refit(self, min_support: float, min_confidence: float):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self._build_model()
