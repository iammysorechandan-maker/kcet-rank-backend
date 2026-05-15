import pandas as pd
from sklearn.isotonic import IsotonicRegression

def get_2026_multiplier(rank):

    # 1 - 3k → almost stable
    if rank <= 3000:
        return 1.012

    # 3k - 10k → slight inflation
    elif rank <= 10000:
        return 1.035

    # 10k - 15k → stronger inflation
    elif rank <= 15000:
        return 1.06

    # 15k - 25k → mild stabilization
    elif rank <= 25000:
        return 1.018

    # 25k - 40k → moderate inflation
    elif rank <= 40000:
        return 1.04

    # 40k - 70k → stronger inflation
    elif rank <= 70000:
        return 1.075

    # 70k - 100k → heavy inflation
    elif rank <= 100000:
        return 1.11

    # 100k+ → progressive inflation
    else:
        extra = min((rank - 100000) / 100000, 0.10)
        return 1.13 + extra


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

        multiplier = get_2026_multiplier(rank_2025)

        rank_2026 = int(rank_2025 * multiplier)

        print("Rank:", rank_2025)
        print("Multiplier:", multiplier)
        print("2026:", rank_2026)

        print("------------")
        print("2025:", rank_2025)
        print("Multiplier:", multiplier)
        print("2026:", rank_2026)
        print("------------")

        return {
            "rank_2025": rank_2025,
            "rank_2026": rank_2026
        }
