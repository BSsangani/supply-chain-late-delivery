# Supply Chain Late Delivery Analysis & Prediction

Analysis of **180,519 order lines (54,854 orders)** from the DataCo Smart Supply Chain dataset to find why orders arrive late and to flag high-risk orders at the moment they are placed.

**Tools:** Python (Pandas, Scikit-learn, XGBoost, Matplotlib, Seaborn), SQL (SQLite, window functions)

## Key results
| KPI | Value |
|---|---|
| Orders analysed (Jan 2015 – Sep 2017, excl. cancelled/fraud) | 54,854 orders / 164,572 lines |
| Late delivery rate | **57.3%** |
| Average promised vs actual shipping time | 2.93 vs 3.5 days |
| First Class late rate | **100%** (promised 1 day, always takes 2) |
| Second Class late rate | 79.8% (promised 2 days, takes 4 on average) |
| Standard Class late rate | 39.8% |

**Root cause:** late delivery is driven by **unrealistic promised lead times on premium shipping modes**, not by market or region (all five markets sit between 56.8% and 57.8% late).

**Prediction model** (trained on older orders, tested on the most recent 20%, using only information known when the order is placed):

| Model | ROC-AUC | Accuracy |
|---|---|---|
| Majority-class baseline | – | 0.570 |
| Logistic Regression | 0.746 | 0.705 |
| **XGBoost** | **0.780** | **0.719** |

Orders the model ranks in the **highest-risk 20% were actually late 97.9% of the time**, compared with 32.6% for the lowest-risk 20%. The operations team could use this to prioritise or re-quote the riskiest orders.

## Recommendations
1. Reset the promised lead time for First Class (1 → 2 days) and Second Class (2 → 4 days) to match actual performance — this alone would remove most "late" orders and improve customer trust.
2. Use the risk score at order entry to flag high-risk orders for proactive customer updates.
3. Track on-time delivery weekly by shipping mode as a core KPI.

## Data quality steps
- Dropped personal data columns (names, emails, passwords, street) — GDPR data minimisation.
- Excluded 7,754 cancelled / suspected-fraud lines and the incomplete Oct 2017 – Jan 2018 period.
- Found 86.2% of Order Zipcode values missing, so that field was not used.
- Avoided data leakage: actual shipping days and delivery status were excluded from the model.

## Files
- `Supply_Chain_Late_Delivery.ipynb` — full analysis with outputs
- `sql/kpi_queries.sql` — KPI queries
- `charts/` — charts used in the analysis

## Data
DataCo Smart Supply Chain for Big Data Analysis — Constante, Silva & Pereira (2019), Mendeley Data, CC BY 4.0: https://data.mendeley.com/datasets/8gx2fvg2k6/5 .
Download `DataCoSupplyChainDataset.csv` into `data/` before running (the file is 95 MB, so it is not stored in this repository).
