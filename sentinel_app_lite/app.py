
import os
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="SENTINEL", layout="wide")

@st.cache_data
def load_file(name):
    app_folder = os.path.dirname(os.path.abspath(__file__))
    return pd.read_csv(os.path.join(app_folder, "data", name))

region_results = load_file("region_results.csv")
country_results = load_file("country_results.csv")
monthly = load_file("monthly_region.csv")
monthly["month"] = pd.to_datetime(monthly["month"])
summary = load_file("summary.csv").set_index("item")["value"]
last_month = pd.to_datetime(summary["last_complete_month"])
next_month = last_month + pd.DateOffset(months=1)

MODEL = "claude-haiku-4-5-20251001"

def get_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return None

def ask_claude(system_prompt, user_text, max_tokens):
    import anthropic
    client = anthropic.Anthropic(api_key=get_api_key())
    reply = client.messages.create(model=MODEL, max_tokens=max_tokens, system=system_prompt,
                                   messages=[{"role": "user", "content": user_text}])
    return reply.content[0].text

READ_PROMPT = '''You read a short news report about armed conflict and return ONLY a JSON object with these fields:
"event_type": one of ["Battles", "Explosions/Remote violence", "Violence against civilians", "Protests", "Riots", "Strategic developments"],
"location": place named or "unknown", "deaths_mentioned": a number or null, "tone": "calm", "tense" or "violent",
"summary": one plain-English sentence, "confidence": a number from 0 to 1.
Use only what the text says. Do not guess.'''

EXPLAIN_PROMPT = '''You explain conflict-monitoring results to a non-technical NGO coordinator.
Write 3 to 4 short sentences in plain English.
Use ONLY the numbers given. Do not add facts, causes or names that are not in the data.
Say clearly if the trend might just be random noise.
End with one cautious sentence: this is decision support and local knowledge matters.'''

def read_report(text):
    raw = ask_claude(READ_PROMPT, text, 400)
    start = raw.find("{")
    end = raw.rfind("}") + 1
    return json.loads(raw[start:end])

def explain_row(row):
    facts = json.dumps(row.to_dict(), default=str)
    return ask_claude(EXPLAIN_PROMPT, "Region data: " + facts, 300)

def explain_template(row):
    if "risk_group" in row:
        text = str(row["admin1"]) + " is in the risk group '" + row["risk_group"] + "' with a severity score of " + str(row["severity"]) + " out of 100. "
    else:
        text = str(row["country"]) + ": " + str(row["pct_regions_high_intensity"]) + "% of its regions are high intensity, and the average regional severity is " + str(row["avg_region_severity"]) + " out of 100. "
    text = text + "Last month there were " + str(int(row["last_month"])) + " recorded events. "
    text = text + "The forecast for next month is about " + str(int(round(row["forecast"]))) + " events (" + row["direction"] + "). "
    if row["trend_is_real"] == "Yes":
        text = text + "This trend is statistically clear. "
    else:
        text = text + "This trend is not statistically clear, so it may be random noise: please confirm with local contacts. "
    text = text + "This is decision support only; local knowledge should guide any decision."
    return text

st.sidebar.title("SENTINEL")
st.sidebar.caption("Signal Extraction & Explainable Intelligence for Conflict Monitoring")
country_list = sorted(country_results["country"])
country = st.sidebar.selectbox("1. Pick a conflict", country_list)

in_country = region_results[region_results["country"] == country]
in_country = in_country.sort_values("severity", ascending=False)
region_options = ["(whole country)"] + list(in_country["admin1"])
region = st.sidebar.selectbox("2. Pick a region (optional)", region_options)

st.sidebar.info("Fixed snapshot: ACLED data up to " + last_month.strftime("%B %Y") +
                ". Forecast is for " + next_month.strftime("%B %Y") + ".")
st.sidebar.caption("Decision support only. It does not replace human judgment.")

if region == "(whole country)":
    row = country_results[country_results["country"] == country].iloc[0]
    history = monthly[monthly["country"] == country].groupby("month")["total_events"].sum().reset_index()
    title = country
else:
    row = in_country[in_country["admin1"] == region].iloc[0]
    history = monthly[(monthly["country"] == country) & (monthly["admin1"] == region)]
    title = region + ", " + country

tab1, tab3, tab4, tab5 = st.tabs(["Results", "Model comparison", "Statistical tests", "Paste a report"])

with tab1:
    st.header(title)

    col1, col2, col3, col4 = st.columns(4)
    if region == "(whole country)":
        col1.metric("Regions that are high intensity", str(row["pct_regions_high_intensity"]) + "%")
        col2.metric("Average regional severity (0-100)", row["avg_region_severity"])
    else:
        col1.metric("Risk group", row["risk_group"])
        col2.metric("Severity score (0-100)", row["severity"])
    change = row["forecast"] - row["last_month"]
    col3.metric("Forecast for " + next_month.strftime("%b %Y") + " (events)", round(row["forecast"]),
                delta=str(round(change)) + " vs last month", delta_color="off")
    col4.metric("Direction", row["direction"], help="Forecast compared with the average of the 6 months before the latest 3.")

    if row["trend_is_real"] == "Yes":
        st.success("Statistical check: the 12-month trend is REAL (p = " + str(row["trend_p"]) + "), not just noise.")
    else:
        st.warning("Statistical check: the 12-month trend is NOT CLEAR (p = " + str(row["trend_p"]) + "). It could be random ups and downs.")

    recent_history = history.sort_values("month").tail(36)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=recent_history["month"], y=recent_history["total_events"],
                                mode="lines+markers", name="Recorded events"))
    figure.add_trace(go.Scatter(x=[next_month], y=[row["forecast"]], mode="markers", name="Forecast",
                                marker=dict(size=12, color="crimson")))
    figure.update_layout(title="Events per month (last 36 months) and next-month forecast", height=380)
    st.plotly_chart(figure, width="stretch")
    st.caption("Rough 95% range for the forecast: " + str(row["low"]) + " to " + str(row["high"]) + " events.")

    st.subheader("Plain-English explanation")
    if st.button("Explain this result"):
        if get_api_key():
            try:
                st.write(explain_row(row))
                st.caption("Written by the LLM (Claude Haiku).")
            except Exception as error:
                st.warning("The LLM call failed (" + type(error).__name__ + "), so here is the template version instead.")
                st.write(explain_template(row))
        else:
            st.write(explain_template(row))
            st.caption("Written by a fixed template (no API key set).")

    if region == "(whole country)":
        st.subheader("Most severe regions in this country")
        table = in_country[["admin1", "severity", "risk_group", "last_month", "forecast", "direction"]].head(10)
        st.dataframe(table, hide_index=True, width="stretch")

    st.subheader("What kinds of events happened (last 12 months)?")
    type_columns = ["battles", "explosions", "civilians", "protests", "riots", "strategic"]
    if region == "(whole country)":
        pick = monthly[monthly["country"] == country]
    else:
        pick = monthly[(monthly["country"] == country) & (monthly["admin1"] == region)]
    pick = pick.sort_values("month").groupby("month")[type_columns].sum().tail(12)
    st.bar_chart(pick.sum())

with tab3:
    st.header("How well do the methods work? (tested on months they never saw)")

    st.subheader("1. Severity: one decision tree against a random forest")
    st.dataframe(load_file("severity_models.csv"), hide_index=True, width="stretch")
    st.caption("R2: 1 is perfect, 0 is no better than guessing the average. Average error is in log units (lower is better). A single tree can memorise the training months; a forest of many trees averages out its mistakes.")
    importance = load_file("feature_importance.csv").set_index("feature")
    st.bar_chart(importance)
    st.caption("This chart is the 'explainable' part: it shows which inputs push the severity estimate up.")

    st.subheader("2. Risk groups: K-means")
    st.dataframe(load_file("cluster_silhouette.csv"), hide_index=True, width="stretch")
    st.caption("We tried 2 to 5 groups. Silhouette: higher means better separated groups. 3 scored best, but below 0.5 means the groups overlap, so read them as rough categories.")
    st.dataframe(load_file("cluster_profile.csv"), hide_index=True, width="stretch")

    st.subheader("3. Forecast: SMA against a simple yardstick")
    st.dataframe(load_file("forecast_backtest.csv"), hide_index=True, width="stretch")
    st.caption("Error as a % of events over a 6-month backtest (lower is better). The moving average beats 'same as last month', but is still wrong by about 13%.")

with tab4:
    st.header("Is it signal or noise?")
    st.write("A trend only counts as real when its p-value is below 0.05.")
    st.dataframe(load_file("stat_tests.csv"), hide_index=True, width="stretch")
    st.caption("trend_p: Spearman trend over the last 12 months. level_change_p: last 6 months against the 6 before (Mann-Whitney).")

with tab5:
    st.header("Paste a news report")
    st.write("The LLM reads the text and pulls out structured details. It does NOT change the forecast.")
    if not get_api_key():
        st.info("No API key found. Add ANTHROPIC_API_KEY in Streamlit Secrets to use this tab. (The 'Explain this result' button on the Results tab still works without a key, using a template.)")

    examples = {"(write my own)": "",
                "Example 1 (made up)": "Armed men attacked a village near the border on Tuesday, killing at least eight civilians, local officials said.",
                "Example 2 (made up)": "Thousands of people marched through the capital on Saturday to demand lower fuel prices. The demonstration ended peacefully."}
    choice = st.selectbox("Try an example", list(examples.keys()))
    text = st.text_area("News text", value=examples[choice], height=160)

    if st.button("Extract details"):
        if text.strip() == "":
            st.warning("Please paste some text first.")
        elif not get_api_key():
            st.warning("No API key found.")
        else:
            try:
                result = read_report(text)
                a, b, c = st.columns(3)
                a.metric("Event type", result.get("event_type", "?"))
                b.metric("Tone", result.get("tone", "?"))
                c.metric("Confidence", result.get("confidence", "?"))
                st.write("Summary: " + str(result.get("summary", "")))
                st.write("Location: " + str(result.get("location", "unknown")))
                st.write("Deaths mentioned: " + str(result.get("deaths_mentioned")))
                if result.get("confidence", 1) < 0.6:
                    st.warning("Low confidence: please check this one by hand.")
            except Exception as error:
                st.error("The LLM call failed: " + type(error).__name__ + ": " + str(error)[:150])
