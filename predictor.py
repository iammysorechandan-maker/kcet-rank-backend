import pandas as pd
from sklearn.isotonic import IsotonicRegression

class RankPredictor:

    def __init__(self, csv_path):

        raw = pd.read_csv(csv_path)

        raw = raw.sort_values("aggregate")

        grouped = raw.groupby(
            "aggregate",
            as_index=False
        )["rank"].median()

        self.x = grouped["aggregate"]
        self.y = grouped["rank"]

        self.model = IsotonicRegression(
            increasing=False,
            out_of_bounds="clip"
        )

        self.model.fit(self.x, self.y)

    def predict(self, aggregate):

        rank_2025 = int(
            self.model.predict([aggregate])[0]
        )

        rank_2026 = int(rank_2025 * 0.985)

        return {
            "rank_2025": rank_2025,
            "rank_2026": rank_2026
        }