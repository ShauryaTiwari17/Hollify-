import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="WeatherTrip", page_icon="🌤️", layout="wide")
st.title("🌤️ WeatherTrip")
st.caption("Find a holiday destination that fits your weather, travel and food preferences.")

@st.cache_data
def load_data():
    return pd.read_excel("weather_travel_recommendation_dataset.xlsx", sheet_name="TravelWeatherData")

@st.cache_resource
def train_model():
    data = load_data()
    # Train to classify the weather of an observation, not to predict a destination.
    features = ["Avg_Temp_C", "Avg_Humidity_Pct", "Rainfall_mm"]
    X, y = data[features], data["Weather_Category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy, features

try:
    df = load_data()
    model, accuracy, features = train_model()
except Exception as exc:
    st.error(f"Could not load the dataset or train the model: {exc}")
    st.stop()

st.subheader("Your holiday preferences")
with st.form("preferences"):
    a, b, c = st.columns(3)
    with a:
        month = st.selectbox("When will you travel?", [
            "January", "February", "March", "April", "May", "June", "July",
            "August", "September", "October", "November", "December"
        ], index=11)
        temp_pref = st.selectbox("Preferred weather", ["Cold", "Cool", "Warm", "Hot"])
    with b:
        dislike = st.selectbox("What bothers you most?", ["Humidity", "Rain", "Heat", "Cold", "Nothing"])
        trip_pref = st.selectbox("Trip type", ["Any", "Mountains", "Beach", "Nature", "City", "Heritage"])
    with c:
        cuisine = st.selectbox("Preferred cuisine", ["Any", "Punjabi", "North Indian", "South Indian", "Himachali", "Maharashtrian", "Goan", "Seafood", "Assamese", "Odia", "Rajasthani", "Kumaoni", "Garhwali", "Karnataka", "Konkani", "Mughlai", "North Eastern"])
        diet = st.selectbox("Diet", ["Either", "Vegetarian", "Non-Vegetarian"])
    submitted = st.form_submit_button("Find my destinations", type="primary", use_container_width=True)

# A transparent weighted content-based ranking layer.
# Score values are 0..1 and are not calibrated probabilities.
def weather_match(temp, preference):
    centers = {"Cold": 8, "Cool": 18, "Warm": 26, "Hot": 33}
    return max(0.0, 1.0 - abs(temp - centers[preference]) / 20.0)

def discomfort_match(row, dislike):
    if dislike == "Humidity":
        return float(np.clip((85 - row.Avg_Humidity_Pct) / 45, 0, 1))
    if dislike == "Rain":
        return float(np.clip(1 - row.Rainfall_mm / 250, 0, 1))
    if dislike == "Heat":
        return float(np.clip((34 - row.Avg_Temp_C) / 22, 0, 1))
    if dislike == "Cold":
        return float(np.clip((row.Avg_Temp_C - 5) / 25, 0, 1))
    return 1.0

def trip_match(actual, wanted):
    if wanted == "Any":
        return 1.0
    return 1.0 if wanted.lower() in actual.lower() else 0.0

def cuisine_match(row, wanted):
    if wanted == "Any":
        return 1.0
    cuisines = [row.Primary_Cuisine, row.Secondary_Cuisine, row.Common_Alternate_Cuisine]
    if wanted.lower() == str(cuisines[0]).lower():
        return 1.0
    return 0.8 if any(wanted.lower() == str(x).lower() for x in cuisines[1:]) else 0.0

def diet_match(row, wanted):
    if wanted == "Either":
        return 1.0
    available = str(row.Diet_Availability)
    if wanted == "Vegetarian":
        return 1.0 if "Veg" in available else 0.0
    return 1.0 if "Non-Veg" in available else 0.0

if submitted:
    month_df = df[df["Month"] == month].copy()
    # Group repeated synthetic observations into one monthly profile per destination.
    numeric = month_df.groupby("Destination", as_index=False)[features].mean()
    meta_cols = ["Destination", "State", "Trip_Type", "Primary_Cuisine", "Secondary_Cuisine", "Common_Alternate_Cuisine", "Diet_Availability"]
    meta = month_df[meta_cols].drop_duplicates(subset="Destination")
    candidates = numeric.merge(meta, on="Destination")
    candidates["Predicted_Weather"] = model.predict(candidates[features])
    scores = []
    for row in candidates.itertuples(index=False):
        weather = weather_match(row.Avg_Temp_C, temp_pref)
        comfort = discomfort_match(row, dislike)
        trip = trip_match(row.Trip_Type, trip_pref)
        food = cuisine_match(row, cuisine)
        dietary = diet_match(row, diet)
        # Weather remains the biggest component; food and trip type matter too.
        score = 0.35 * weather + 0.20 * comfort + 0.20 * trip + 0.20 * food + 0.05 * dietary
        scores.append(score)
    candidates["Match_Score"] = np.round(np.array(scores) * 100, 1)
    candidates = candidates.sort_values("Match_Score", ascending=False)

    st.divider()
    st.subheader(f"Best matches for {month}")
    st.caption("Scores are heuristic preference-match points, NOT prediction probabilities. Weather profiles are synthetic monthly averages.")
    for idx, row in enumerate(candidates.head(3).itertuples(index=False), 1):
        with st.container(border=True):
            st.markdown(f"### {idx}. {row.Destination}, {row.State} — {row.Match_Score:.1f}% match")
            st.write(f"**Random Forest weather category:** {row.Predicted_Weather}")
            st.write(f"🌡️ {row.Avg_Temp_C:.1f}°C  |  💧 {row.Avg_Humidity_Pct:.0f}% humidity  |  🌧️ {row.Rainfall_mm:.0f} mm rainfall")
            st.write(f"**Trip:** {row.Trip_Type}  |  **Food:** {row.Primary_Cuisine}, {row.Secondary_Cuisine}, {row.Common_Alternate_Cuisine}")
            st.write(f"**Diet options:** {row.Diet_Availability}")

    st.subheader("Compare destination match scores")
    st.bar_chart(candidates.set_index("Destination")["Match_Score"])
    best = candidates.iloc[0]["Destination"]
    st.subheader(f"Historical-pattern view: {best}")
    city_data = df[df["Destination"] == best]
    monthly = city_data.groupby("Month")[features].mean().reindex([
        "January", "February", "March", "April", "May", "June", "July", "August",
        "September", "October", "November", "December"
    ])
    chart_metric = st.selectbox("Explore weather trends", features, format_func=lambda x: {
        "Avg_Temp_C": "Average temperature (°C)",
        "Avg_Humidity_Pct": "Average humidity (%)",
        "Rainfall_mm": "Rainfall (mm)"
    }[x])
    st.line_chart(monthly[chart_metric])
    with st.expander("See all destination scores"):
        st.dataframe(candidates[["Destination", "State", "Predicted_Weather", "Match_Score"]], hide_index=True, use_container_width=True)

with st.expander("How this project works / limitations"):
    st.markdown(f"""
    **Workflow:** Synthetic weather observations → Random Forest weather classifier → monthly destination profiles → weighted preference ranking → food suggestions + charts.

    **Model holdout accuracy:** {accuracy:.1%}. This is measured on *synthetic, rule-derived labels* and does **not** establish real-world forecasting accuracy.

    **Important:** Weather categories were generated from temperature, humidity and rainfall rules in the synthetic dataset. The forest learns to reproduce these rules; it does not forecast future weather. Cuisine availability is a rough illustrative label, not a verified restaurant inventory. Replace the synthetic observations with real public weather records for a public-weather-data assignment.
    """)
