"""
Synthetic Online Shopper Session Data Generator
================================================
Reproducible synthetic data generator for the shopper intervention ML pipeline.
Statistically and behaviorally consistent with the UCI Online Shoppers Intention dataset.

Usage:
    python generate_shopper_data.py                  # 12,330 rows (default), saves to CSV
    python generate_shopper_data.py --n 5000         # Custom row count
    python generate_shopper_data.py --n 1000 --seed 99 --out my_data.csv
    python generate_shopper_data.py --preview        # Print first 5 rows to stdout

    # In Python:
    from generate_shopper_data import generate_shopper_data
    df = generate_shopper_data(n=5000, seed=42)
"""

import argparse
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def generate_shopper_data(n: int = 12330, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic online shopper session dataset.

    Parameters
    ----------
    n    : Number of sessions (rows) to generate.
    seed : Random seed for full reproducibility.

    Returns
    -------
    pd.DataFrame with the same schema as the UCI Online Shoppers Intention dataset.
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Assign intent tiers
    #    ~55% low-intent, ~30% medium-intent, ~15% high-intent
    # ------------------------------------------------------------------
    intent_probs = [0.55, 0.30, 0.15]
    intent = rng.choice(["low", "medium", "high"], size=n, p=intent_probs)
    low    = intent == "low"
    med    = intent == "medium"
    high   = intent == "high"

    # ------------------------------------------------------------------
    # 2. Visitor type  (influences engagement independently)
    # ------------------------------------------------------------------
    visitor_type = rng.choice(
        ["Returning_Visitor", "New_Visitor", "Other"],
        size=n,
        p=[0.68, 0.24, 0.08],
    )
    returning = visitor_type == "Returning_Visitor"
    new_vis   = visitor_type == "New_Visitor"

    # ------------------------------------------------------------------
    # 3. Session activity counts
    # ------------------------------------------------------------------
    # Administrative  (many zeros, skewed)
    admin_base = np.where(low, 0.4, np.where(med, 1.5, 3.5))
    Administrative = rng.poisson(admin_base).clip(0, 20).astype(int)
    # Returning visitors browse admin pages more
    Administrative = np.where(returning, (Administrative * rng.uniform(1.0, 1.5, n)).astype(int), Administrative).clip(0, 20)

    # Informational (mostly 0–2)
    info_base = np.where(low, 0.2, np.where(med, 0.6, 1.2))
    Informational = rng.poisson(info_base).clip(0, 10).astype(int)

    # ProductRelated (core engagement signal)
    pr_low  = rng.integers(1, 6,   size=n)
    pr_med  = rng.integers(5, 31,  size=n)
    pr_high = rng.integers(30, 201, size=n)
    ProductRelated = np.where(low, pr_low, np.where(med, pr_med, pr_high))
    ProductRelated = np.where(returning, (ProductRelated * rng.uniform(1.0, 1.3, n)).astype(int), ProductRelated).clip(1, 200)

    # ------------------------------------------------------------------
    # 4. Durations (right-skewed, correlated with counts)
    # ------------------------------------------------------------------
    def _duration(count, scale_per_page, max_val, rng, size):
        base = count * rng.exponential(scale_per_page, size=size)
        noise = rng.exponential(scale_per_page * 0.3, size=size)
        dur = np.where(count == 0, 0.0, base + noise)
        return dur.clip(0, max_val)

    Administrative_Duration   = _duration(Administrative,   120, 3000,  rng, n)
    Informational_Duration    = _duration(Informational,     80, 2000,  rng, n)
    ProductRelated_Duration   = _duration(ProductRelated,    55, 10000, rng, n)

    # ------------------------------------------------------------------
    # 5. Behavioral rates
    # ------------------------------------------------------------------
    # BounceRates: high for low-intent / new visitors, low for high-intent
    bounce_mean = np.where(low, 0.55, np.where(med, 0.30, 0.08))
    bounce_mean = np.where(new_vis, bounce_mean * 1.3, bounce_mean).clip(0, 1)
    BounceRates = rng.beta(
        a=np.where(bounce_mean > 0.5, 1.2, 2.0),
        b=np.where(bounce_mean > 0.5, 1.0, 4.0),
    ).clip(0, 1)

    # ExitRates >= BounceRates on average
    ExitRates = (BounceRates + rng.beta(1.5, 5, size=n) * 0.25).clip(0, 1)

    # PageValues: strongly 0 for low-intent, skewed high for high-intent
    pv_low  = np.zeros(n)
    pv_med  = rng.exponential(3, size=n).clip(0, 30)
    pv_high = rng.exponential(25, size=n).clip(0, 150)
    PageValues = np.where(low, pv_low, np.where(med, pv_med, pv_high))
    # Sprinkle zeros into medium-intent (partial engagement)
    zero_mask = (med) & (rng.random(n) < 0.45)
    PageValues = np.where(zero_mask, 0, PageValues)

    # ------------------------------------------------------------------
    # 6. Temporal / contextual features
    # ------------------------------------------------------------------
    months_ordered = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    # Seasonal weights: higher traffic Nov/Dec/May
    month_weights = np.array([0.04, 0.04, 0.06, 0.07, 0.10, 0.07, 0.05, 0.05, 0.07, 0.08, 0.18, 0.19])
    month_weights /= month_weights.sum()
    Month = rng.choice(months_ordered, size=n, p=month_weights)

    # SpecialDay: spikes in Nov/Dec and around holidays
    holiday_months = {"Nov", "Dec", "Feb", "May"}
    special_base = np.where(np.isin(Month, list(holiday_months)), 0.35, 0.05)
    SpecialDay = np.where(
        rng.random(n) < special_base,
        rng.choice([0.2, 0.4, 0.6, 0.8, 1.0], size=n, p=[0.3, 0.3, 0.2, 0.15, 0.05]),
        0.0,
    )

    Weekend = rng.random(n) < 0.23  # ~23% of sessions on weekends

    # ------------------------------------------------------------------
    # 7. Categorical browser/OS/region/traffic features
    # ------------------------------------------------------------------
    OperatingSystems = rng.choice(range(1, 9),  size=n, p=[0.28, 0.22, 0.20, 0.12, 0.07, 0.05, 0.04, 0.02])
    Browser          = rng.choice(range(1, 14), size=n, p=[0.30, 0.20, 0.15, 0.10, 0.07, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01])
    Region           = rng.choice(range(1, 10), size=n, p=[0.25, 0.15, 0.14, 0.12, 0.10, 0.09, 0.07, 0.05, 0.03])
    TrafficType      = rng.choice(range(1, 21), size=n, p=np.array([
        15, 12, 10, 9, 8, 7, 6, 5, 4, 4, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1
    ], dtype=float) / 100)

    # ------------------------------------------------------------------
    # 8. Revenue (target variable)
    # ------------------------------------------------------------------
    # Base probability by intent tier
    rev_prob = np.where(low, 0.02, np.where(med, 0.12, 0.52))

    # Modifiers
    rev_prob += np.where(returning, 0.04, 0.0)
    rev_prob += np.where(new_vis,  -0.02, 0.0)
    rev_prob += (PageValues / 200) * 0.25          # high PageValues -> purchase
    rev_prob -= BounceRates * 0.15                 # high bounce -> less likely
    rev_prob += np.where(Weekend, 0.01, 0.0)
    rev_prob += SpecialDay * 0.04
    rev_prob += np.where(np.isin(Month, ["Nov", "Dec"]), 0.05, 0.0)

    rev_prob = rev_prob.clip(0.01, 0.95)
    Revenue = rng.random(n) < rev_prob

    # ------------------------------------------------------------------
    # 9. Add natural noise / outliers
    # ------------------------------------------------------------------
    # ~1% extreme high-engagement outlier sessions
    outlier_mask = rng.random(n) < 0.01
    ProductRelated          = np.where(outlier_mask, rng.integers(150, 201, n), ProductRelated)
    ProductRelated_Duration = np.where(outlier_mask, rng.uniform(8000, 10000, n), ProductRelated_Duration)
    PageValues              = np.where(outlier_mask, rng.uniform(80, 150, n),     PageValues)

    # ------------------------------------------------------------------
    # 10. Assemble DataFrame
    # ------------------------------------------------------------------
    df = pd.DataFrame({
        "Administrative":           Administrative.astype(int),
        "Administrative_Duration":  Administrative_Duration.round(2),
        "Informational":            Informational.astype(int),
        "Informational_Duration":   Informational_Duration.round(2),
        "ProductRelated":           ProductRelated.astype(int),
        "ProductRelated_Duration":  ProductRelated_Duration.round(2),
        "BounceRates":              BounceRates.round(6),
        "ExitRates":                ExitRates.round(6),
        "PageValues":               PageValues.round(2),
        "SpecialDay":               SpecialDay.round(2),
        "Month":                    Month,
        "OperatingSystems":         OperatingSystems.astype(int),
        "Browser":                  Browser.astype(int),
        "Region":                   Region.astype(int),
        "TrafficType":              TrafficType.astype(int),
        "VisitorType":              visitor_type,
        "Weekend":                  Weekend,
        "Revenue":                  Revenue,
    })

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic online shopper session data for ML pipelines."
    )
    parser.add_argument("--n",       type=int,   default=12330,                   help="Number of rows to generate (default: 12330)")
    parser.add_argument("--seed",    type=int,   default=42,                      help="Random seed (default: 42)")
    parser.add_argument("--out",     type=str,   default="shopper_synthetic.csv", help="Output CSV filename")
    parser.add_argument("--preview", action="store_true",                         help="Print first 5 rows instead of saving")
    args = parser.parse_args()

    print(f"Generating {args.n:,} sessions (seed={args.seed})...")
    df = generate_shopper_data(n=args.n, seed=args.seed)

    revenue_rate = df["Revenue"].mean()
    print(f"  Revenue rate : {revenue_rate:.1%}  ({df['Revenue'].sum():,} / {len(df):,} sessions)")
    print(f"  Visitor mix  : {df['VisitorType'].value_counts().to_dict()}")

    if args.preview:
        print(df.head().to_string())
    else:
        df.to_csv(args.out, index=False)
        print(f"  Saved -> {args.out}")


if __name__ == "__main__":
    main()
