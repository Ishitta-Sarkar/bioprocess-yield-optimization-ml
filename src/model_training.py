import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


dataset = pd.read_csv("data/bioprocess_dataset.csv")

X = dataset.drop(columns=["run_id", "product_yield_g_l"])
y = dataset["product_yield_g_l"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Bioprocess Yield Prediction Model")
print("=" * 40)
print("Mean Absolute Error:", round(mae, 3))
print("R² Score:", round(r2, 3))

results = pd.DataFrame([
    {
        "model": "Random Forest Regressor",
        "mean_absolute_error": round(mae, 3),
        "r2_score": round(r2, 3)
    }
])

results.to_csv("results/model_performance.csv", index=False)
