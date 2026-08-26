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
| `filter_price_min` | float64 | 393 | raw filter → distance |
| `filter_price_max` | float64 | 1 167 | raw filter → distance |
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
- `price_distance_min = price - filter_price_min` and `price_distance_max = filter_price_max - price` are **always ≥ 0** and **do vary**. They are the only non-degenerate filter-derived signals: position of price inside the selected band.

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
- Titles look like catalog strings with brand, product, pack size, and a parenthetical quality cue.
- Descriptions are **highly templated** (“for online grocery orders. Listed under …”). Signal may be weak; ablation (tabular-only) is how we find out.

## Physical volume

`dimensions_in` looks like `3.3 x 4.0 x 4.1"`. A three-float `L x W x H` parse succeeds on **all 10 000** rows. Product of sides: median ~109 in³, heavy right tail (max ~3404). Scale after median impute.

## Time

Timestamps span **2024-07-08 → 2026-07-08** (UTC). Hour-of-day and day-of-week are roughly uniform. Purchase rate varies modestly (hour ~0.10–0.17, Sunday slightly higher). Cyclical sin/cos is justified; raw hour integers are not.

## `package_size`

27 string levels that restate size + unit already covered by `net_weight_oz` and `unit_of_measure`. **Not used.**

## `nutrition_score`

Range 0–99; **1 244 zeros**. Could be a true score or a sentinel. Default: treat as a real number (no recoding) until EDA plots in the notebook suggest otherwise ([OPEN_DECISIONS.md](OPEN_DECISIONS.md) D8).
