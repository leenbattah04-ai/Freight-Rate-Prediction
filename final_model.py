import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# ============================================================
# FINAL MODEL — CATBOOST + ROUTE + DATE FEATURES
# ============================================================

TRAIN_PATH = "data/train-test.csv"
VALIDATION_PATH = "data/validation.csv"
DECEMBER_PATH = "data/december_chart_inputs.csv"

OUTPUT_VALIDATION = "validation_predictions.csv"
OUTPUT_DECEMBER = "data/december_chart_inputs.csv"

TARGET = "posted_rate"

BASE_NUMERICAL_FEATURES = [
    "distance",
    "weight",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "market_index",
    "quote_signal",
]

DATE_FEATURES = [
    "month",
    "day_of_week",
    "day",
]

NUMERICAL_FEATURES = BASE_NUMERICAL_FEATURES + DATE_FEATURES

CATEGORICAL_FEATURES = [
    "pickup",
    "delivery",
    "equipment",
    "route",
]

FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

FINAL_ITERATIONS = 165


def add_common_features(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="raise")

    if "weight" in df.columns:
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce").abs()

    df["pickup"] = df["pickup"].fillna("UNKNOWN").astype(str)
    df["delivery"] = df["delivery"].fillna("UNKNOWN").astype(str)

    df["route"] = df["pickup"] + " -> " + df["delivery"]

    # Date features selected using temporal validation.
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day"] = df["date"].dt.day

    return df


print("=" * 70)
print("FINAL FREIGHT RATE PREDICTION MODEL")
print("CATBOOST + ROUTE + DATE FEATURES")
print("=" * 70)

# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)
validation_df = pd.read_csv(VALIDATION_PATH)

train_df = add_common_features(train_df)
validation_df = add_common_features(validation_df)

print(f"\nTraining data:   {train_df.shape}")
print(f"Validation data: {validation_df.shape}")

# ============================================================
# TRAINING-DERIVED FALLBACK STATISTICS
# ============================================================

# Global numerical medians are learned from training only.
train_medians = train_df[NUMERICAL_FEATURES].median()

# City coordinates are learned from training only.
pickup_coordinates = (
    train_df.groupby("pickup")[["pickup_lat", "pickup_lon"]]
    .median()
)

delivery_coordinates = (
    train_df.groupby("delivery")[["delivery_lat", "delivery_lon"]]
    .median()
)

# Route-specific market signals are learned from training only.
route_signal_medians = (
    train_df.groupby("route")[["market_index", "quote_signal"]]
    .median()
)

# ============================================================
# PREPROCESS TRAIN / VALIDATION
# ============================================================

train_df[NUMERICAL_FEATURES] = (
    train_df[NUMERICAL_FEATURES].fillna(train_medians)
)

validation_df[NUMERICAL_FEATURES] = (
    validation_df[NUMERICAL_FEATURES].fillna(train_medians)
)

for col in CATEGORICAL_FEATURES:
    train_df[col] = train_df[col].fillna("UNKNOWN").astype(str)
    validation_df[col] = validation_df[col].fillna("UNKNOWN").astype(str)

# ============================================================
# TRAIN FINAL MODEL
# ============================================================

X_train = train_df[FEATURES].copy()
y_train = train_df[TARGET].copy()
X_validation = validation_df[FEATURES].copy()

model = CatBoostRegressor(
    loss_function="RMSE",
    iterations=FINAL_ITERATIONS,
    learning_rate=0.05,
    depth=7,
    l2_leaf_reg=5,
    random_seed=42,
    verbose=False,
)

model.fit(
    X_train,
    y_train,
    cat_features=CATEGORICAL_FEATURES,
    verbose=False,
)

# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

validation_predictions = model.predict(X_validation)
validation_predictions = np.maximum(validation_predictions, 0)
validation_predictions = np.nan_to_num(
    validation_predictions,
    nan=0.0,
    posinf=1e9,
    neginf=0.0,
)

submission = pd.DataFrame({
    "load_id": validation_df["load_id"],
    "predicted_rate": validation_predictions,
})

submission.to_csv(OUTPUT_VALIDATION, index=False)

print(f"\nSaved: {OUTPUT_VALIDATION}")
print("\nValidation prediction statistics:")
print(submission["predicted_rate"].describe())

# ============================================================
# DECEMBER PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("DECEMBER PREDICTIONS")
print("=" * 70)

# Auto-detect comma/semicolon so this works with the original
# template and with a previously completed CSV.
december_df = pd.read_csv(DECEMBER_PATH, sep=None, engine="python")

if len(december_df) != 31:
    raise ValueError(
        f"Expected 31 December rows, but found {len(december_df)}"
    )

# The scorer requires one row for every day in December.
# If date is absent in the original template, create it.
if "date" not in december_df.columns:
    december_df["date"] = pd.date_range(
        start="2025-12-01",
        end="2025-12-31",
        freq="D",
    )

december_df = add_common_features(december_df)

# Recover city coordinates from training data rather than using
# unrelated global coordinate medians.
for col in ["pickup_lat", "pickup_lon"]:
    if col not in december_df.columns:
        december_df[col] = december_df["pickup"].map(
            pickup_coordinates[col]
        )
    else:
        missing = december_df[col].isna()
        december_df.loc[missing, col] = (
            december_df.loc[missing, "pickup"]
            .map(pickup_coordinates[col])
        )

for col in ["delivery_lat", "delivery_lon"]:
    if col not in december_df.columns:
        december_df[col] = december_df["delivery"].map(
            delivery_coordinates[col]
        )
    else:
        missing = december_df[col].isna()
        december_df.loc[missing, col] = (
            december_df.loc[missing, "delivery"]
            .map(delivery_coordinates[col])
        )

# market_index and quote_signal are not supplied in the fixed
# December inputs. Use training-only route medians when the route
# was observed; otherwise fall back to the global training median.
for col in ["market_index", "quote_signal"]:
    route_values = december_df["route"].map(route_signal_medians[col])

    if col not in december_df.columns:
        december_df[col] = route_values
    else:
        december_df[col] = december_df[col].fillna(route_values)

# Final fallback for any unresolved numerical value.
december_df[NUMERICAL_FEATURES] = (
    december_df[NUMERICAL_FEATURES].fillna(train_medians)
)

for col in CATEGORICAL_FEATURES:
    december_df[col] = (
        december_df[col].fillna("UNKNOWN").astype(str)
    )

X_december = december_df[FEATURES].copy()

december_predictions = model.predict(X_december)
december_predictions = np.maximum(december_predictions, 0)
december_predictions = np.nan_to_num(
    december_predictions,
    nan=0.0,
    posinf=1e9,
    neginf=0.0,
)

# Keep exactly the seven columns required by score.py.
december_output = pd.DataFrame({
    "pickup": december_df["pickup"],
    "delivery": december_df["delivery"],
    "distance": december_df["distance"],
    "equipment": december_df["equipment"],
    "weight": december_df["weight"],
    "date": december_df["date"].dt.strftime("%Y-%m-%d"),
    "predicted_rate": december_predictions,
})

december_output.to_csv(OUTPUT_DECEMBER, index=False)

print(f"\nSaved: {OUTPUT_DECEMBER}")
print("\nDecember prediction statistics:")
print(december_output["predicted_rate"].describe())

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.get_feature_importance(),
}).sort_values("importance", ascending=False)

print("\n" + "=" * 70)
print("FINAL MODEL FEATURE IMPORTANCE")
print("=" * 70)
print(importance.to_string(index=False))

print("\n" + "=" * 70)
print("FINAL MODEL COMPLETE")
print("=" * 70)

print(
    "\nNext step:\n"
    "python score.py "
    "--predictions validation_predictions.csv "
    "--december-predictions data/december_chart_inputs.csv"
)
