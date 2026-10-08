# WeatherTrip — Weather + Food Travel Recommender

## Run
1. Install Python 3.11 or newer.
2. Open a terminal inside this folder.
3. Run `pip install -r requirements.txt`
4. Run `streamlit run app.py`

## What happens
1. Read the included synthetic weather-and-destination Excel dataset.
2. Train a Random Forest to classify weather categories from temperature, humidity and rainfall.
3. Ask the user six quick preference questions.
4. Average observations for each destination in the chosen month.
5. Use a transparent weighted content-based score to rank destinations (35% temperature, 20% weather discomfort, 20% trip type, 20% cuisine, 5% diet).
6. Display top destinations, food options and weather charts.

## Scientific limitations
- **All weather values are synthetic**, with small random perturbations. They are not real meteorological observations.
- `Weather_Category` was derived from weather features by fixed rules, so the classifier only imitates these rules; high test accuracy does not mean weather forecasting skill.
- Cuisine, dietary availability and travel categories are approximate example metadata, not verified for individual restaurants.
- Match percentages are **heuristic scores**, not calibrated likelihoods.
- To satisfy an assignment explicitly requiring *public* weather data, replace the weather rows with a real dataset and document its source.
