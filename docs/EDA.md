# Data profile

Computed from `data/raw/supermarket_products.csv` (not assumed from the PDF). All design choices that depend on “what the table actually contains” should point here.

## Shape and columns

- **10 000 rows**, **22 columns**, **2 012 distinct `query_id`**.
- Impressions per query: mean ~4.97, min 1, max 8.
- No missing values except **`allergens` (4 455 / 44.55%)**.

| Column | Dtype | Distinct | Role |
| --- | --- | --- | --- |
| `title` | object | 9 910 | text |
| `description` | object | 9 112 | text |
| `ingredients` | object | 190 | text |
| `price` | float64 | 2 135 | numeric |
| `category` | object | 12 | categorical |
| `timestamp` | object (ISO UTC) | 9 999 | cyclical time |
| `query_id` | object | 2 012 | **split grouping only** |
| `filter_category` | object | 12 | raw filter (see degeneracy) |
| `filter_price_min` | float64 | 393 | raw filter → `relative_price_position` |
| `filter_price_max` | float64 | 1 167 | raw filter → `relative_price_position` |
| `filter_storage_type` | object | 3 | raw filter (see degeneracy) |
| `cart` | bool | 2 | **leakage — drop** |
| `bought` | bool | 2 | **target** |
| `brand` | object | 15 | categorical |
| `package_size` | object | 27 | unused (redundant) |
| `unit_of_measure` | object | 5 | categorical |
| `net_weight_oz` | float64 | 3 753 | numeric |
| `dimensions_in` | object | 9 864 | parse → `volume` |
| `storage_type` | object | 3 | categorical |
| `allergens` | object | 7 + NaN | categorical |
| `nutrition_score` | int64 | 83 | numeric |
| `country_of_origin` | object | 10 | categorical |

## Target and imbalance

- `bought=False`: 8 699 (86.99%)
- `bought=True`: 1 301 (13.01%)

Primary metric is **PR-AUC** because the positive class is the minority and BTR is about ranking likely purchases among impressions.

Queries with zero purchases: **52.6%**. Queries with more than one purchase: **14.1%**. Mean purchases per query: **0.65**.

## Leakage: `cart`

|  | `bought=False` | `bought=True` |
| --- | ---: | ---: |
| `cart=False` | 6 993 | **0** |
| `cart=True` | 1 706 | 1 301 |

- \(P(bought \mid cart=\text{false}) = 0\)
- \(P(bought \mid cart=\text{true}) \approx 0.43\)

`cart` is a **post-impression action** on the path to purchase. Using it would change the task from BTR (purchase among **impressions**) to conversion among cart-adds. **Drop as a feature.** See [adr/0002](../adr/0002-leakage-cart-query-id.md).

## `query_id` is a search, not a feature

Filters (`filter_category`, both price bounds) are **constant within a query**. Timestamps are **not**: almost every row has its own timestamp (mean distinct timestamps per query ≈ rows per query).

`query_id` must **not** enter the model (high-cardinality identifier → memorization). It **must** drive the split so impressions from the same search do not leak across train/valid/test. See [adr/0005](../adr/0005-grouped-splits.md).

## Degenerate filter relationships (critical)

On this file:

- `category == filter_category` for **100%** of rows
- `storage_type == filter_storage_type` for **100%** of rows
- `filter_price_min ≤ price ≤ filter_price_max` for **100%** of rows

Implications:

- `is_category_match` and `is_storage_match` are **constant 1**. They encode the intended *interaction idea* but carry **zero variance** here.
- Raw `filter_category` / `filter_storage_type` would duplicate `category` / `storage_type` one-hots. Dropping raw `filter_*` after engineering is correct.
- Absolute band distances (`price - filter_price_min`, `filter_price_max - price`) vary but are collinear with band width; the useful signal is **where** price sits inside the band.
- **Only filter-derived feature kept:** `relative_price_position = (price - filter_price_min) / (filter_price_max - filter_price_min)` (already in \([0, 1]\) on this CSV; guard zero-width bands). Do **not** feed the two raw distances.

Recommended handling is in [OPEN_DECISIONS.md](OPEN_DECISIONS.md) D1 and [adr/0003](../adr/0003-feature-engineering.md).

## Categorical supports

- **category / filter_category:** Pantry, Produce, Beverages, Dairy, Frozen, Bakery, Snacks, Meat, Household, Personal Care, Seafood, Baby.
- **storage_type / filter_storage_type:** Ambient (5 524), Refrigerated (2 911), Frozen (1 565).
- **unit_of_measure:** oz, lb, fl oz, ct, gal.
- **country_of_origin:** United States (7 500) + nine others (~245–331 each).
- **brand:** 15 labels, reasonably balanced (~597–726 rows).
- **allergens (non-null):** Wheat, Milk, Soy, Tree nuts, Peanuts, Shellfish, Fish. Treat NaN as an explicit level (e.g. `None`).

## Text

Concatenated `title + description + ingredients` is short:

- Whitespace tokens: mean **~40**, max **50**.
- BERT WordPiece, including `[CLS]`/`[SEP]` and two `[SEP]` field separators: mean **55.6**, p95 **64**, **max 74**. See D10.
- Titles look like catalog strings with brand, product, pack size, and a parenthetical cue.
- Descriptions are **highly templated** (“for online grocery orders. Listed under …”).

### The text carries almost all the signal — via a behavioural cue (critical)

Measured on a grouped 70/15/15 split, seed 42:

| Model | ROC-AUC | PR-AUC | Baseline |
| --- | ---: | ---: | ---: |
| Logistic regression, full 62-dim tabular block | 0.517 | 0.141 | 0.137 |
| Gradient boosting, full 62-dim tabular block | 0.551 | 0.154 | 0.137 |
| TF-IDF on concatenated text | 0.955 | 0.648 | 0.132 |
| **Title parenthetical alone, mapped to its train purchase rate** | **0.960** | **0.670** | 0.132 |

The whole tabular pipeline is close to noise. The title parenthetical takes 20 values in three regimes:

| Cue group | n | Purchase rate |
| --- | ---: | ---: |
| Customer Favorite, Best Seller, Top Rated, #1 Pick | 1 931 | 0.63 – 0.68 |
| Well Reviewed, Shopper Favorite, Highly Rated, Popular Choice | 1 973 | 0.02 – 0.04 |
| Remaining 12 (Clearance Listing, Rarely Reordered, Standard Listing, …) | 6 096 | 0.00 |

Top TF-IDF n-grams: `most repurchased`, `frequently reordered`, `returning customers`, `customer pick`. Bottom: `less repurchased`, `rarely reordered`, `limited`. `description` restates the same behaviour, so deleting the title parenthetical and the final description sentence still leaves ROC-AUC **0.953** — the cue is redundantly encoded across both fields.

This is a target-derived proxy visible at render time, not `cart`-style leakage. Handling — keep it, and ablate against a cue-stripped arm — is [adr/0009](../adr/0009-text-behavioural-cue.md).

### Metric noise floor

Across 20 grouped seeds at 85/15, a near-null tabular model gives test ROC-AUC 0.571 ± 0.019 and PR-AUC 0.160 ± 0.016 (range 0.129 – 0.193). Held-out prevalence itself varies 0.118 – 0.153 (sd 0.009). **Differences below ~0.05 PR-AUC are not resolvable on a single split**, and because PR-AUC's baseline *is* the prevalence, valid and test PR-AUC are not directly comparable unless splits are prevalence-stratified. See D15 and D16.

## Physical volume

`dimensions_in` looks like `3.3 x 4.0 x 4.1"`. A three-float `L x W x H` parse succeeds on **all 10 000** rows. Product of sides: median ~109 in³, heavy right tail (max ~3404).

Raw `volume` has skew **2.97**; after `StandardScaler` its max \(|z|\) is **9.61** and **50 rows** exceed \(|z| > 5\). Applying `log1p` first brings skew to **0.44** and max \(|z|\) to **3.05**. The other three numerics need no transform (max \(|z|\): `price` 4.68, `net_weight_oz` 4.36, `nutrition_score` 1.96). Decision: **`log1p` on `volume` only**, then median impute and scale (D17).

## Time

Timestamps span **2024-07-08 → 2026-07-08** (UTC). Hour-of-day and day-of-week are roughly uniform. Purchase rate varies modestly (hour ~0.10–0.17, Sunday slightly higher). Cyclical sin/cos is justified; raw hour integers are not.

## `package_size`

27 string levels that restate size + unit already covered by `net_weight_oz` and `unit_of_measure`. **Not used.**

## `nutrition_score`

Range 0–99; **1 244 zeros**. Could be a true score or a sentinel. Default: treat as a real number (no recoding) until EDA plots in the notebook suggest otherwise ([OPEN_DECISIONS.md](OPEN_DECISIONS.md) D8).
