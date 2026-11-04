"""Generate small synthetic events file for CI, unit testing, and make smoke."""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_smoke_events(out_path: Path, n_users: int = 200, seed: int = 42):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    user_ids = [1000 + i for i in range(n_users)]
    categories = [
        (2053013555631882655, "electronics.smartphone"),
        (2053013554415534527, "computers.notebook"),
        (2053013552326770747, "appliances.kitchen.refrigerators"),
        (2053013565488497341, "apparel.shoes"),
        (2053013560530829877, "furniture.bedroom.bed"),
    ]
    brands = ["samsung", "apple", "xiaomi", "huawei", "lg", "unknown"]

    records = []
    # Date range: 2019-10-01 to 2019-10-31
    base_timestamp = pd.Timestamp("2019-10-01 00:00:00")

    for uid in user_ids:
        n_sessions = rng.randint(1, 6)
        for s in range(n_sessions):
            session_id = f"sess_{uid}_{s}"
            sess_time = base_timestamp + pd.Timedelta(
                days=int(rng.randint(0, 30)), hours=int(rng.randint(0, 23))
            )
            n_events = rng.randint(1, 10)

            for e in range(n_events):
                event_time = sess_time + pd.Timedelta(minutes=int(e * 3))
                cid, ccode = categories[rng.randint(0, len(categories))]
                brand = brands[rng.randint(0, len(brands))]
                price = float(round(rng.uniform(10.0, 1000.0), 2))

                # Event type mix
                r = rng.uniform()
                if r < 0.70:
                    etype = "view"
                elif r < 0.90:
                    etype = "cart"
                else:
                    etype = "purchase"

                records.append(
                    {
                        "event_time": event_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "event_type": etype,
                        "product_id": rng.randint(100000, 999999),
                        "category_id": cid,
                        "category_code": ccode,
                        "brand": brand,
                        "price": price,
                        "user_id": uid,
                        "user_session": session_id,
                    }
                )

    df = pd.DataFrame(records)
    # Add a couple duplicate rows to test deduplication
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    df.to_csv(out_path, index=False)
    print(f"Generated smoke events: {len(df):,} rows -> {out_path}")
    return df


if __name__ == "__main__":
    p = Path(__file__).resolve().parent / "smoke_events.csv"
    generate_smoke_events(p)
