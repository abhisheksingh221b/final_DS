import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, silhouette_score

warnings.filterwarnings("ignore")

DATA_FILE = "data/acled_weekly_aggregated.csv"
COUNTRIES = ["Ukraine", "Palestine", "Sudan", "Myanmar", "Syria", "Democratic Republic of Congo", "Yemen"]
TYPE_COLS = {"Battles": "battles", "Explosions/Remote violence": "explosions", "Violence against civilians": "civilians",
             "Protests": "protests", "Riots": "riots", "Strategic developments": "strategic"}
COUNT_COLS = ["battles", "explosions", "civilians", "protests", "riots", "strategic"]

print("1. Loading and cleaning the data")
weekly = pd.read_csv(DATA_FILE)
weekly = weekly[weekly["country"].isin(COUNTRIES)]
weekly["week"] = pd.to_datetime(weekly["week"])
exposure_text = weekly["population_exposure"].astype(str)
exposure_text = exposure_text.str.replace(",", "")
weekly["population_exposure"] = exposure_text.astype(float)
weekly["month"] = weekly["week"].dt.to_period("M").dt.to_timestamp()
weekly = weekly[weekly["week"] >= "2020-01-01"]
newest_month = weekly["month"].max()
weekly = weekly[weekly["month"] < newest_month]
last_month = weekly["month"].max()

print("2. Building the monthly table")
weekly["type_col"] = weekly["event_type"].map(TYPE_COLS)
counts = weekly.pivot_table(index=["country", "admin1", "month"], columns="type_col", values="events",
                            aggfunc="sum", fill_value=0)
for col in COUNT_COLS:
    if col not in counts.columns:
        counts[col] = 0
counts = counts[COUNT_COLS]
group = weekly.groupby(["country", "admin1", "month"])
extra = pd.DataFrame()
extra["fatalities"] = group["fatalities"].sum()
extra["exposure"] = group["population_exposure"].sum()
extra["n_subtypes"] = group["sub_event_type"].nunique()
monthly = counts.join(extra).reset_index()

all_months = pd.date_range(weekly["month"].min(), weekly["month"].max(), freq="MS")
pairs = monthly[["country", "admin1"]].drop_duplicates()
grid = pairs.merge(pd.DataFrame({"month": all_months}), how="cross")
monthly = grid.merge(monthly, on=["country", "admin1", "month"], how="left")
monthly = monthly.fillna(0)
monthly["total_events"] = monthly[COUNT_COLS].sum(axis=1)
monthly["civilian_share"] = 0.0
monthly["unrest_share"] = 0.0
has_events = monthly["total_events"] > 0
monthly.loc[has_events, "civilian_share"] = monthly["civilians"] / monthly["total_events"]
monthly.loc[has_events, "unrest_share"] = (monthly["protests"] + monthly["riots"]) / monthly["total_events"]
monthly = monthly.sort_values(["country", "admin1", "month"]).reset_index(drop=True)

monthly_small = monthly[["country", "admin1", "month", "total_events", "fatalities"] + COUNT_COLS]
monthly_small.to_csv("data/monthly_region.csv", index=False)

print("3. Severity: comparing models")
SPLIT_DATE = pd.Timestamp("2025-06-01")
FEATURES = COUNT_COLS + ["civilian_share", "n_subtypes", "log_exposure"]
active = monthly[monthly["total_events"] > 0].copy()
active["log_exposure"] = np.log1p(active["exposure"])
active["target"] = np.log1p(active["fatalities"])
train = active[active["month"] <= SPLIT_DATE]
test = active[active["month"] > SPLIT_DATE]

models = {"Baseline (always the average)": DummyRegressor(),
          "Decision Tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=42),
          "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=42)}
names_list = []
r2_list = []
error_list = []
for name in models:
    model = models[name]
    model.fit(train[FEATURES], train["target"])
    guess = model.predict(test[FEATURES])
    names_list.append(name)
    r2_list.append(round(r2_score(test["target"], guess), 3))
    error_list.append(round(mean_absolute_error(test["target"], guess), 3))
severity_table = pd.DataFrame({"model": names_list, "R2": r2_list, "average_error": error_list})
severity_table.to_csv("data/severity_models.csv", index=False)

best_name = severity_table.set_index("model")["R2"].idxmax()
final_model = models[best_name]
final_model.fit(active[FEATURES], active["target"])
reference = np.sort(final_model.predict(active[FEATURES]))
importance = pd.DataFrame({"feature": FEATURES, "importance": models["Random Forest"].feature_importances_})
importance.to_csv("data/feature_importance.csv", index=False)
print("   best model:", best_name)

print("4. Risk groups: K-means")
start_of_window = last_month - pd.DateOffset(months=11)
recent = monthly[monthly["month"] >= start_of_window]
sum_columns = COUNT_COLS + ["fatalities", "total_events"]
sums = recent.groupby(["country", "admin1"])[sum_columns].sum()
sums = sums[sums["total_events"] > 0]
feat = pd.DataFrame(index=sums.index)
feat["avg_events"] = sums["total_events"] / 12
feat["deaths_per_event"] = sums["fatalities"] / sums["total_events"]
feat["civilian_share"] = sums["civilians"] / sums["total_events"]
feat["unrest_share"] = (sums["protests"] + sums["riots"]) / sums["total_events"]
feat["avg_deaths"] = sums["fatalities"] / 12
X = feat[["avg_events", "deaths_per_event", "civilian_share", "unrest_share"]].copy()
X["avg_events"] = np.log1p(X["avg_events"])
X["deaths_per_event"] = np.log1p(X["deaths_per_event"])
X_scaled = StandardScaler().fit_transform(X)

k_list = []
silhouette_list = []
for k in range(2, 6):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X_scaled)
    k_list.append(k)
    silhouette_list.append(round(silhouette_score(X_scaled, labels), 3))
pd.DataFrame({"number_of_groups": k_list, "silhouette": silhouette_list}).to_csv("data/cluster_silhouette.csv", index=False)

kmeans_labels = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X_scaled)
feat["cluster"] = kmeans_labels
avg_deaths_by_cluster = feat.groupby("cluster")["avg_deaths"].mean()
order = list(avg_deaths_by_cluster.sort_values().index)
names = {}
names[order[0]] = "Low activity"
names[order[1]] = "Moderate"
names[order[2]] = "High intensity"
feat["risk_group"] = feat["cluster"].map(names)
profile_columns = ["avg_events", "deaths_per_event", "civilian_share", "unrest_share"]
profile = feat.groupby("risk_group")[profile_columns].mean().round(2)
profile["regions"] = feat["risk_group"].value_counts()
profile.reset_index().to_csv("data/cluster_profile.csv", index=False)

print("5. Forecast: SMA against a simple yardstick")

def forecast_naive(values):
    return float(values[-1])

def forecast_sma(values):
    return float(np.mean(values[-3:]))

METHODS = {"Same as last month (yardstick)": forecast_naive, "SMA (average of last 3 months)": forecast_sma}

country_month = monthly.groupby(["country", "month"])[["total_events", "fatalities"]].sum().reset_index()

def get_series(country):
    rows = country_month[country_month["country"] == country].sort_values("month")
    return rows["total_events"].values

error_total = {}
actual_total = {}
for method_name in METHODS:
    error_total[method_name] = 0.0
    actual_total[method_name] = 0.0
for country in COUNTRIES:
    series = get_series(country)
    for method_name in METHODS:
        for i in range(6, 0, -1):
            guess = METHODS[method_name](series[:-i])
            truth = series[-i]
            error_total[method_name] = error_total[method_name] + abs(guess - truth)
            actual_total[method_name] = actual_total[method_name] + truth
method_list = []
error_percent = []
for method_name in METHODS:
    method_list.append(method_name)
    error_percent.append(round(error_total[method_name] / actual_total[method_name] * 100, 1))
backtest = pd.DataFrame({"method": method_list, "error_percent": error_percent})
backtest.to_csv("data/forecast_backtest.csv", index=False)
best_method_name = "SMA"
print("   forecast used:", best_method_name)

def forecast_row(values):
    guess = forecast_sma(values)
    spread = np.std(values[-12:])
    baseline = float(np.mean(values[-9:-3]))
    if baseline >= 3:
        ratio = guess / baseline
    else:
        ratio = 1.0
    if ratio > 1.10:
        direction = "Rising"
    elif ratio < 0.90:
        direction = "Falling"
    else:
        direction = "Stable"
    low = guess - 1.96 * spread
    if low < 0:
        low = 0
    high = guess + 1.96 * spread
    return {"last_month": float(values[-1]), "earlier_avg": round(baseline, 1), "forecast": round(guess, 1),
            "low": round(low, 1), "high": round(high, 1), "direction": direction}

def trend_test(values):
    if values[-12:].std() > 0:
        rho, p = stats.spearmanr(np.arange(12), values[-12:])
    else:
        rho, p = 0, 1
    if p < 0.05:
        verdict = "Yes"
    else:
        verdict = "No"
    return round(rho, 2), round(p, 3), verdict

print("6. Statistical tests")
test_rows = []
for country in COUNTRIES:
    series = get_series(country)
    rho, trend_p, trend_real = trend_test(series)
    before = series[-12:-6]
    recent6 = series[-6:]
    stat, mw_p = stats.mannwhitneyu(recent6, before, alternative="two-sided")
    test_rows.append({"country": country, "trend_rho": rho, "trend_p": trend_p,
                      "trend_is_real": trend_real, "avg_before": round(before.mean()), "avg_recent": round(recent6.mean()),
                      "level_change_p": round(mw_p, 3)})
stat_tests = pd.DataFrame(test_rows)
stat_tests.to_csv("data/stat_tests.csv", index=False)

country_month["deaths_per_event"] = 0.0
busy = country_month["total_events"] > 0
country_month.loc[busy, "deaths_per_event"] = country_month["fatalities"] / country_month["total_events"]

print("7. Putting it together (this can take a minute)")
three_back = last_month - pd.DateOffset(months=2)
last3 = monthly[monthly["month"] >= three_back].copy()
last3["log_exposure"] = np.log1p(last3["exposure"])
last3["estimate"] = final_model.predict(last3[FEATURES])
last3.loc[last3["total_events"] == 0, "estimate"] = 0
avg_estimate = last3.groupby(["country", "admin1"])["estimate"].mean()

rows = []
for (country, region), g in monthly.groupby(["country", "admin1"]):
    values = g.sort_values("month")["total_events"].values
    estimate = avg_estimate[(country, region)]
    below = np.sum(reference < estimate)
    severity = round(below / len(reference) * 100)
    if values[-12:].sum() == 0:
        fc = {"last_month": 0.0, "earlier_avg": 0.0, "forecast": 0.0, "low": 0.0, "high": 0.0, "direction": "Stable"}
    else:
        fc = forecast_row(values)
    rho, trend_p, trend_real = trend_test(values)
    if (country, region) in feat.index:
        risk_group = feat.loc[(country, region), "risk_group"]
    else:
        risk_group = "Inactive"
    row = {"country": country, "admin1": region, "risk_group": risk_group, "severity": severity}
    row.update(fc)
    row["trend_p"] = trend_p
    row["trend_is_real"] = trend_real
    rows.append(row)
results = pd.DataFrame(rows)
results.to_csv("data/region_results.csv", index=False)

country_rows = []
for country in COUNTRIES:
    series = get_series(country)
    row = {"country": country}
    row.update(forecast_row(series))
    rho, trend_p, trend_real = trend_test(series)
    row["trend_p"] = trend_p
    row["trend_is_real"] = trend_real
    country_rows.append(row)
country_results = pd.DataFrame(country_rows)
for_deaths = monthly[monthly["month"] >= start_of_window].groupby("country")[["fatalities", "total_events"]].sum()
country_results["deaths_per_event"] = country_results["country"].map(for_deaths["fatalities"] / for_deaths["total_events"]).round(2)
high_share = []
avg_sev = []
for country in COUNTRIES:
    region_rows = results[results["country"] == country]
    high_share.append(round((region_rows["risk_group"] == "High intensity").mean() * 100))
    avg_sev.append(round(region_rows["severity"].mean()))
country_results["pct_regions_high_intensity"] = high_share
country_results["avg_region_severity"] = avg_sev
country_results.to_csv("data/country_results.csv", index=False)
