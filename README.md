# Freight Rate Prediction

Machine Learning Engineer Assessment for **Spotter**.

This project develops a machine learning model to predict freight posted rates using historical shipment data. The solution includes exploratory data analysis, data preprocessing, temporal validation, model comparison, feature engineering, final model training, and prediction generation.

## Project Structure

```text
Freight_Rate_Prediction/
│
├── data/
│   ├── train-test.csv
│   ├── validation.csv
│   ├── validation-predictions-template.csv
│   └── december_chart_inputs.csv
│
├── scorer_results/
│   └── candidate_december.png
│
├── src/
│   └── eda.py
│
├── final_model.py
├── score.py
├── requirements.txt
├── validation_predictions.csv
└── README.md
```

## Dataset

The labeled development dataset contains **48,000 freight loads** covering January through October 2025.

The final validation dataset contains **12,000 unlabeled loads** covering November and December 2025.

The prediction target is:

```text
posted_rate
```

Key input features include shipment distance, weight, pickup and delivery locations, geographic coordinates, equipment type, market index, quote signal, and date.

## Exploratory Data Analysis

Exploratory analysis was performed to understand data quality, feature relationships, and temporal behavior.

Key findings included:

- Distance was the strongest predictor of `posted_rate`.
- Pearson correlation between distance and posted rate was approximately **0.91**.
- Spearman correlation was approximately **0.98**.
- Missing values were present in `weight` and `market_index`.
- Some shipment weights were negative and were corrected using their absolute values.
- No duplicate records requiring removal were identified.
- Temporal patterns were analyzed by month, day of week, day of month, and overall time trend.
- A directional route feature was created as `pickup -> delivery`.

The EDA implementation is available in:

```text
src/eda.py
```

## Validation Strategy

Because the dataset is chronological, a random train/validation split was avoided.

A temporal validation strategy was used to better simulate prediction on future freight loads.

### Fold 1

```text
Training:   January - August 2025
Validation: September - October 2025
```

### Fold 2

```text
Training:   January - September 2025
Validation: October 2025
```

This approach reduces the risk of temporal leakage and provides a more realistic estimate of future model performance.

## Model Development

Several regression approaches were evaluated.

| Model | Internal Validation MAE |
|---|---:|
| Distance-only baseline | ~194.86 |
| Numerical features | ~194.74 |
| Numerical + categorical features | ~162.22 |
| CatBoost | ~123.69 |
| CatBoost + log target | ~144.10 |
| CatBoost + directional route | ~123.05 |

CatBoost provided the strongest performance among the main tested approaches.

Additional calendar features were then evaluated using the same temporal validation strategy:

```text
month
day_of_week
day
```

Adding these features improved the average internal temporal validation MAE from approximately:

```text
122.79 -> 110.88
```

This represents an improvement of approximately **9.7%**.

## Final Model

The final solution uses:

**CatBoostRegressor + Directional Route + Date Features**

### Numerical Features

```text
distance
weight
pickup_lat
pickup_lon
delivery_lat
delivery_lon
market_index
quote_signal
month
day_of_week
day
```

### Categorical Features

```text
pickup
delivery
equipment
route
```

### Model Configuration

```text
iterations = 165
learning_rate = 0.05
depth = 7
l2_leaf_reg = 5
random_seed = 42
loss_function = RMSE
```

The final model is trained on all **48,000 labeled development rows** before generating the final predictions.

## Data Preprocessing

The preprocessing pipeline includes:

- converting shipment dates to datetime;
- correcting negative weights using absolute values;
- imputing missing numerical values using training-data medians;
- replacing missing categorical values with `UNKNOWN`;
- constructing the directional route feature;
- generating calendar features from the shipment date.

All fallback statistics used for prediction are derived from the labeled training data.

## December Prediction Scenario

The provided December scenario contains 31 rows with fixed shipment characteristics:

```text
Pickup:    Lexington
Delivery:  Fort Wayne
Distance:  360 miles
Equipment: Dry Van
Weight:    32,000 lb
```

Only the date changes from December 1 through December 31, 2025.

For model features not explicitly supplied in the December input:

- pickup and delivery coordinates are recovered from historical training data;
- route-specific training medians are used for `market_index` and `quote_signal` when available;
- global training medians are used as fallback values;
- calendar features are generated from each December date.

The final December predictions range approximately from **$845.74 to $860.14**.

The scorer-generated chart is available at:

```text
scorer_results/candidate_december.png
```

## Installation

Python 3.10+ is recommended.

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the Final Model

From the project root directory, run:

```bash
python final_model.py
```

This generates:

```text
validation_predictions.csv
```

and fills the required predictions in:

```text
data/december_chart_inputs.csv
```

## Validate the Outputs

Run the provided scorer:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

A successful run should validate:

```text
12,000 final predictions
31 fixed December predictions
```

and generate:

```text
scorer_results/candidate_december.png
```

The final hidden validation metrics are calculated by Spotter after submission.

## Final Outputs

The main submission outputs are:

```text
validation_predictions.csv
data/december_chart_inputs.csv
scorer_results/candidate_december.png
```

The final validation prediction file contains exactly:

```text
load_id,predicted_rate
```

## Notes

The reported **MAE of approximately 110.88** is the internal temporal validation result used for model selection. It is not the hidden final score on the 12,000-row validation dataset.

The December scenario should also be interpreted as a fixed-route prediction scenario rather than a full simulation of daily freight-market volatility because future `market_index` and `quote_signal` values are not provided.