# Project 3: Demand Forecasting Pipeline with a Natural-Language Query Layer

**Skill focus:** SQL, time-series forecasting, rolling-origin evaluation, experiment tracking, pipeline design, practical GenAI
**Estimated time:** about 16 working days at 2-3 hours/day
**Difficulty:** Intermediate

---

## 1. Business framing

> *"Production and inventory planning depend on knowing how much of each product will be demanded, by region, weeks ahead. Overproduction ties up cash and stock; underproduction loses sales. Can we build a reliable forecasting pipeline on top of a SQL database, and let a non-technical planner ask questions about the results in plain English?"*

**Be transparent about the proxy.** Michelin's real demand data is not public. You will use the M5 Walmart retail dataset as a proxy: it has the same structural challenges (many products, many locations, seasonality, promotions, intermittent demand). State this in the README.

## 2. What you will learn and demonstrate

| Skill | Where it shows up |
|---|---|
| SQL | Loading, aggregating, window functions on a real database |
| Forecasting | Naive baselines, statistical models, LightGBM with lag features |
| Honest evaluation | Rolling-origin cross-validation, scale-free metrics |
| MLOps habits | MLflow experiment tracking, reproducible pipeline |
| Practical GenAI | Guard-railed text-to-SQL or RAG, with failure analysis |
| Communication | Planner-facing insights |

## 3. Dataset

**M5 Forecasting (Walmart), on Kaggle.** Key files:
- `sales_train_validation.csv` (or `_evaluation`): daily unit sales per item-store, in wide format (one column per day, about 1,900 days)
- `calendar.csv`: dates, weekday, events, SNAP (food assistance) flags
- `sell_prices.csv`: weekly price per item-store

Scale: about 30,000 item-store series across 10 stores in 3 states. **That is large for a laptop.** Reduce scope deliberately:
- Choose **one category** (for example FOODS or HOBBIES) or one state, and/or
- Forecast at the **store x department** level (aggregate items), which gives a few dozen series, and keep a smaller item-level subset for a stretch goal.

Document your scoping decision and why. Knowing how to scope is a hiring signal.

**Optional external features:** public macro indicators (for example inflation, consumer confidence) or weather. Only add what you can justify and join without leakage.

## 4. Repository structure

```
demand-forecasting/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── warehouse.db            # SQLite database
├── sql/
│   ├── 01_schema.sql
│   ├── 02_exploration.sql
│   └── 03_features.sql
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   └── 03_models.ipynb
├── src/
│   ├── load_data.py
│   ├── features.py
│   ├── backtest.py             # rolling-origin evaluation
│   ├── models.py
│   └── nlq/
│       ├── schema_prompt.py
│       ├── sql_guard.py
│       └── assistant.py
├── app/
│   └── streamlit_app.py
├── mlruns/                      # MLflow (gitignored)
├── reports/
└── tests/
```

## 5. Step-by-step plan

### Phase 0: Setup (Day 1)

- Environment: pandas, numpy, scikit-learn, lightgbm, statsmodels, statsforecast (or prophet), mlflow, streamlit, sqlalchemy, plus an LLM client library.
- GitHub repo, `.gitignore` (data, mlruns, API keys). **Never commit API keys.** Use environment variables or a `.env` file excluded from git.

### Phase 1: Build the data warehouse in SQL (Days 2-3)

Convert the wide sales file to long format and load into SQLite (upgrade to PostgreSQL later if you want):

```python
import pandas as pd, sqlite3

sales = pd.read_csv("data/raw/sales_train_validation.csv")
id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
long = sales.melt(id_vars=id_cols, var_name="d", value_name="units")
cal = pd.read_csv("data/raw/calendar.csv")
prices = pd.read_csv("data/raw/sell_prices.csv")

con = sqlite3.connect("data/warehouse.db")
long.to_sql("sales", con, if_exists="replace", index=False)
cal.to_sql("calendar", con, if_exists="replace", index=False)
prices.to_sql("prices", con, if_exists="replace", index=False)
```

Filter to your chosen scope **before** melting so memory does not explode. Add indexes on `(store_id, item_id, d)` and on `calendar(d)`.

**SQL practice that doubles as interview preparation.** Write and save these in `sql/02_exploration.sql`:
- Total units by category and state
- Top 10 items per store using `ROW_NUMBER() OVER (PARTITION BY store_id ORDER BY total DESC)`
- 7-day moving average of sales with `AVG(units) OVER (... ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)`
- Week-over-week growth using `LAG()`
- Share of zero-sales days per item (intermittency)
- Revenue by joining sales, calendar, and prices

### Phase 2: EDA (Day 4)

Questions to answer with plots:
- Weekly and yearly seasonality? Day-of-week pattern?
- Effect of events and holidays?
- How intermittent are item-level series (many zeros)? This decides which models make sense.
- How do price changes relate to units sold?

### Phase 3: Baselines and the evaluation framework (Days 5-6)

**Build the evaluation framework before the models.** Use **rolling-origin (time-series) cross-validation**:
- Fix a horizon (for example 28 days, as in M5).
- Create 3-5 folds: train up to date T, forecast T+1 to T+28, then move T forward.
- Never shuffle. Never train on the future.

Metrics:
- **MAE** and **RMSE** (scale-dependent)
- **MASE or RMSSE** (scaled against a naive forecast, so series with different volumes are comparable)
- **Bias** (average signed error), because systematic under- or over-forecasting has distinct business consequences
- Optionally weighted RMSSE as used in M5

Baselines (each must beat a lazy approach to be worth it):
1. **Naive**: last value
2. **Seasonal naive**: value from 7 days ago (or same weekday last week average)
3. **Moving average**

Report these as the table every later model must beat.

### Phase 4: Feature engineering (Days 7-8)

Build features in SQL and/or pandas. The leakage rule: a feature for day *t* may only use information available **before** day *t* (shifted).

```python
df = df.sort_values(["series_id", "date"])
g = df.groupby("series_id")["units"]
for lag in (7, 14, 28):
    df[f"lag_{lag}"] = g.shift(lag)
for w in (7, 28):
    df[f"roll_mean_{w}"] = g.transform(lambda x: x.shift(28).rolling(w).mean())
```

Note the `shift(28)` before rolling: if you forecast 28 days ahead in one shot, features must be lagged at least by the horizon (or you must forecast recursively).

Feature ideas:
- Calendar: weekday, month, week of year, is_weekend, event flags
- Price: current price, price relative to the item's average, recent price change
- Lags and rolling statistics (as above)
- Series identifiers as categorical (store, department)

### Phase 5: Models (Days 9-11)

Compare:

| Model | Notes |
|---|---|
| Seasonal naive | Your baseline |
| ETS / ARIMA (via statsforecast) | Strong statistical benchmark per series |
| Prophet | Handles holidays and seasonality easily; trains per series |
| **LightGBM (global model)** | One model across all series with lag and calendar features, typically the strongest in practice |
| (Optional) ensemble | Average of LightGBM and a statistical model |

Why a **global** model matters: with thousands of series, training one model per series is slow and fragile; a global model learns shared patterns. Be ready to explain this trade-off.

For intermittent item-level demand, try a Tweedie or Poisson objective in LightGBM and explain why (non-negative, many zeros).

**Track every experiment in MLflow:** model type, features used, horizon, parameters, CV metrics. Compare runs in the MLflow UI and include a screenshot in the README.

### Phase 6: Evaluation and business interpretation (Day 12)

- Show CV metrics per fold and averaged (variance across folds matters).
- Break results down by category, store, and volume tier. Where does the model fail (new items, promotions, low-volume items)?
- Plot actual vs. forecast for 5-6 representative series including a bad one.
- Translate into business terms: for example, "using forecast X instead of seasonal naive reduces forecast error by N%, which would mean roughly M fewer units of safety stock *under simplified assumptions*." Label every assumption.

### Phase 7: Natural-language query layer (Days 13-15)

**Goal:** a planner types "Which departments are forecast to grow most next month in Texas?" and gets an answer backed by real SQL results.

Store your forecasts in a `forecasts` table in the database (series id, date, forecast, model name, run id).

Architecture (text-to-SQL):
1. Build a **schema prompt**: table names, column names, short descriptions, a few example rows, and 3-5 example question/SQL pairs.
2. Send the question plus schema to an LLM and ask for **a single SQLite SELECT query only**.
3. **Validate the generated SQL (guardrails)** before running:
   - Allow only `SELECT` statements; reject anything containing `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `ATTACH`, `PRAGMA`
   - Allow only whitelisted tables
   - Open the database connection **read-only**
   - Enforce a `LIMIT` and a query timeout
4. Execute, then have the LLM summarize the returned rows in plain English, restating which table and filters were used so the user can verify.
5. On error, return a polite failure message; optionally allow one automatic retry with the error message fed back.

```python
import re, sqlite3

FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|attach|pragma|create|replace)\b", re.I)

def validate_sql(sql: str, allowed_tables: set[str]) -> str:
    sql = sql.strip().rstrip(";")
    if not sql.lower().startswith("select") or FORBIDDEN.search(sql) or ";" in sql:
        raise ValueError("Only a single read-only SELECT is allowed")
    # check referenced tables against the allow-list, then enforce a LIMIT
    return sql if re.search(r"\blimit\b", sql, re.I) else sql + " LIMIT 200"

con = sqlite3.connect("file:data/warehouse.db?mode=ro", uri=True)  # read-only
```

(Regex checks are a first line of defence, not a complete one. The read-only connection is the real protection. Say so in your write-up.)

**Evaluate it like a data scientist, not a demo-maker:**
- Write **30-50 test questions** with known correct SQL or answers (easy aggregations, joins, time filters, ambiguous questions, unanswerable questions, adversarial ones like "delete all forecasts").
- Measure **execution accuracy** (does it return the correct result?) and the share of unsafe queries blocked.
- Categorize failures (wrong join, wrong date handling, hallucinated column) and report them honestly.
- Optionally compare prompt variants (with and without example pairs) and show the effect.

If you prefer a RAG variant: index your forecast reports and model documentation and let users ask about methodology. Text-to-SQL is recommended because it exercises SQL skills too.

Keep this module modest. A small, well-evaluated assistant beats a flashy, untested chatbot.

### Phase 8: App and packaging (Day 16)

Streamlit app with:
- A forecast explorer (select store/department, see history plus forecast)
- A model comparison tab (metrics from MLflow or saved CSV)
- A chat box for the NL query layer that **shows the generated SQL** to the user

Provide a `Makefile` or simple script sequence: `load_data` → `build_features` → `train` → `backtest` → `write_forecasts` → `run_app`. Reproducibility is part of the deliverable.

## 6. Common pitfalls

- Random train/test split on time series.
- Features that peek into the future (rolling windows without shifting).
- Evaluating only on one time window, which could be unrepresentative.
- Ignoring intermittent demand and judging with MAPE (undefined with zeros; use MASE/RMSSE).
- Trying to run the full M5 on a laptop and never finishing. Scope first.
- Letting the LLM run unchecked SQL, or committing API keys.
- Showing LLM answers without ever measuring accuracy.

## 7. Stretch goals

- Hierarchical reconciliation: make store, state, and total forecasts consistent.
- Probabilistic forecasts (quantile LightGBM) and use the 90th percentile to set safety stock.
- Orchestrate the pipeline with Prefect or Airflow-style scheduling.
- Add a data-drift check and an automated retraining trigger.
- Dockerize the app.

## 8. Definition of done

- [ ] SQL warehouse with documented schema and saved analytical queries
- [ ] Rolling-origin backtest framework
- [ ] 3+ baselines/statistical models and a global LightGBM compared fairly
- [ ] MLflow tracking with screenshot
- [ ] Error breakdown and business interpretation with stated assumptions
- [ ] NL query layer with guardrails and a documented evaluation set
- [ ] Streamlit app and a one-command pipeline
- [ ] README with results and limitations

## 9. Resume bullets (adapt with your real numbers)

- Built an end-to-end demand forecasting pipeline on a SQL warehouse (M5 retail data as a proxy), where a global LightGBM model cut RMSSE by **X%** versus a seasonal-naive baseline under 4-fold rolling-origin validation.
- Tracked all experiments in MLflow and diagnosed errors by category, store, and volume tier.
- Developed a guard-railed text-to-SQL assistant for planners, reaching **Y%** execution accuracy on a 40-question test set while blocking all tested unsafe queries.

## 10. Interview questions to prepare

1. Why rolling-origin CV instead of k-fold?
2. Why a global model instead of one model per series, and what are the downsides?
3. Why avoid MAPE here? What did you use?
4. How did you prevent leakage in your lag and rolling features?
5. How did you make the text-to-SQL layer safe, and how did you measure its accuracy?
6. If Michelin gave you real data, what additional problems would you expect (new products, stockouts censoring demand, promotions, hierarchical reporting)?
