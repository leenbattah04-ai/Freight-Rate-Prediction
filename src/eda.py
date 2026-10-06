from __future__ import annotations

from pathlib import Path
import pandas as pd


# PROJECT PATHS

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"


# LOAD DATA

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / "train-test.csv")
    validation = pd.read_csv(DATA_DIR / "validation.csv")

    return train, validation


# DATE / TIME ANALYSIS

def analyze_date_time(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("DATE / TIME ANALYSIS")
    print("=" * 70)


    train = train.copy()
    validation = validation.copy()

    train["date"] = pd.to_datetime(train["date"])
    validation["date"] = pd.to_datetime(validation["date"])


    print("\n" + "-" * 70)
    print("DATE RANGES")
    print("-" * 70)

    print("\nTrain:")
    print(f"First date: {train['date'].min().date()}")
    print(f"Last date:  {train['date'].max().date()}")
    print(f"Number of unique dates: {train['date'].nunique()}")

    print("\nValidation:")
    print(f"First date: {validation['date'].min().date()}")
    print(f"Last date:  {validation['date'].max().date()}")
    print(f"Number of unique dates: {validation['date'].nunique()}")


    train["month"] = train["date"].dt.month
    validation["month"] = validation["date"].dt.month

    train["month_name"] = train["date"].dt.strftime("%B")
    validation["month_name"] = validation["date"].dt.strftime("%B")

    print("\n" + "-" * 70)
    print("MONTHLY DATA DISTRIBUTION")
    print("-" * 70)

    train_monthly = (
        train.groupby(["month", "month_name"])
        .size()
        .reset_index(name="count")
        .sort_values("month")
    )

    validation_monthly = (
        validation.groupby(["month", "month_name"])
        .size()
        .reset_index(name="count")
        .sort_values("month")
    )

    print("\nTrain:")
    print(train_monthly.to_string(index=False))

    print("\nValidation:")
    print(validation_monthly.to_string(index=False))


    print("\n" + "-" * 70)
    print("MONTHLY POSTED RATE")
    print("-" * 70)

    monthly_target = (
        train.groupby(["month", "month_name"])["posted_rate"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max",
        )
        .reset_index()
        .sort_values("month")
    )

    print(monthly_target.to_string(index=False))


    print("\n" + "-" * 70)
    print("MONTHLY MARKET INDEX")
    print("-" * 70)

    monthly_market = (
        train.groupby(["month", "month_name"])["market_index"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max",
        )
        .reset_index()
        .sort_values("month")
    )

    print(monthly_market.to_string(index=False))


    print("\n" + "-" * 70)
    print("MONTHLY QUOTE SIGNAL")
    print("-" * 70)

    monthly_quote = (
        train.groupby(["month", "month_name"])["quote_signal"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max",
        )
        .reset_index()
        .sort_values("month")
    )

    print(monthly_quote.to_string(index=False))


    print("\n" + "-" * 70)
    print("MONTHLY EQUIPMENT DISTRIBUTION")
    print("-" * 70)

    monthly_equipment = (
        pd.crosstab(
            train["month_name"],
            train["equipment"],
            normalize="index",
        )
        * 100
    )

    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    existing_months = [
        month for month in month_order
        if month in monthly_equipment.index
    ]

    monthly_equipment = monthly_equipment.loc[existing_months]

    print("Percent of each equipment type within each month:")
    print(monthly_equipment.round(2).to_string())


    print("\n" + "-" * 70)
    print("DAY OF WEEK ANALYSIS")
    print("-" * 70)

    train["day_of_week"] = train["date"].dt.day_name()

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    weekday_target = (
        train.groupby("day_of_week")["posted_rate"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
        )
        .reindex(weekday_order)
    )

    print(weekday_target.to_string())


    print("\n" + "-" * 70)
    print("DAY OF MONTH ANALYSIS")
    print("-" * 70)

    train["day_of_month"] = train["date"].dt.day

    day_of_month_target = (
        train.groupby("day_of_month")["posted_rate"]
        .agg(
            count="count",
            mean="mean",
            median="median",
        )
        .sort_index()
    )

    print(day_of_month_target.to_string())


    print("\n" + "-" * 70)
    print("TIME TREND")
    print("-" * 70)

    daily_target = (
        train.groupby("date")["posted_rate"]
        .agg(
            count="count",
            mean="mean",
            median="median",
        )
        .reset_index()
        .sort_values("date")
    )

    print("\nFirst 10 days:")
    print(daily_target.head(10).to_string(index=False))

    print("\nLast 10 days:")
    print(daily_target.tail(10).to_string(index=False))


    print("\n" + "-" * 70)
    print("EARLY vs LATE PERIOD")
    print("-" * 70)

    train["period"] = train["date"].dt.to_period("M")

    period_summary = (
        train.groupby("period")["posted_rate"]
        .agg(
            count="count",
            mean="mean",
            median="median",
        )
        .reset_index()
    )

    print(period_summary.to_string(index=False))


    print("\n" + "-" * 70)
    print("TRAIN / VALIDATION DATE OVERLAP")
    print("-" * 70)

    train_dates = set(train["date"].dt.date)
    validation_dates = set(validation["date"].dt.date)

    overlapping_dates = train_dates.intersection(validation_dates)
    validation_only_dates = validation_dates - train_dates
    train_only_dates = train_dates - validation_dates

    print(f"Unique train dates: {len(train_dates)}")
    print(f"Unique validation dates: {len(validation_dates)}")
    print(f"Overlapping dates: {len(overlapping_dates)}")
    print(f"Validation-only dates: {len(validation_only_dates)}")
    print(f"Train-only dates: {len(train_only_dates)}")


    print("\n" + "-" * 70)
    print("VALIDATION MONTH COVERAGE")
    print("-" * 70)

    validation_month_range = (
        validation.groupby(["month", "month_name"])
        .size()
        .reset_index(name="count")
        .sort_values("month")
    )

    print(validation_month_range.to_string(index=False))


# NUMERICAL FEATURE RELATIONSHIPS

def analyze_numerical_target_relationships(train: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("NUMERICAL FEATURES → TARGET RELATIONSHIP")
    print("=" * 70)

    numerical_features = [
        "distance",
        "weight",
        "market_index",
        "quote_signal",
    ]

    available_features = [
        feature for feature in numerical_features
        if feature in train.columns
    ]

    print("\nPearson correlation with posted_rate:")

    correlations = (
        train[available_features + ["posted_rate"]]
        .corr(numeric_only=True)["posted_rate"]
        .drop("posted_rate")
        .sort_values(ascending=False)
    )

    print(correlations.to_string())


    print("\n" + "-" * 70)
    print("SPEARMAN CORRELATION WITH POSTED RATE")
    print("-" * 70)

    spearman = (
        train[available_features + ["posted_rate"]]
        .corr(method="spearman", numeric_only=True)["posted_rate"]
        .drop("posted_rate")
        .sort_values(ascending=False)
    )

    print(spearman.to_string())


# MAIN

def main():
    train, validation = load_data()

    analyze_date_time(train, validation)
    analyze_numerical_target_relationships(train)


if __name__ == "__main__":
    main()
