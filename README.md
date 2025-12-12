# Estimating Delivery Time with Confidence: A Practical and Explainable ETA Prediction System

Food-delivery operations deal with constant variabilities from multiple restaurants in order to prepare at different speeds, couriers move through inconsistent traffic, and weather conditions shift unexpectedly. Yet customers expect a delivery time that is both accurate and dependable.

Our goal is straightforward: to quantify how long orders truly take, predict ETA with consistent accuracy, evaluate how reliable those predictions are under different conditions, and explain the operational factors that influence delivery time.

The project focuses on three core tasks:
1. Producing a consistently accurate ETA,
2. Measuring how often predictions fall within an acceptable tolerance window,
3. Explaining why delivery times increase or vary across conditions.

The Streamlit dashboard consolidates all components of the workflow data quality checks, model comparisons, tolerance based accuracy, feature importance, and segmented level diagnostics, hence teams can understand not only what the ETA is, but also why the estimate behaves the way it does. This information may provides a foundation for every operational decisions especially in term of making more stable customer-facing promises, and ongoing improvement of delivery performance.

# Understanding Our Deliveries

Each row in our dataset is a story:
- Who is involved?
- What is happening?
- Where does it take place?
- When does it occur?
- Why does it matter?
- How does it unfold?
Before any machine learning, we enrich, clean, and make sense of the columns so the results remain trustworthy.

| Column                   | Type        | Meaning                                                  | Notes & Handling                                                                                |
| ------------------------ | ----------- | -------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `order_id`               | `str`       | Unique identifier for each order                         | Useful for tracing rows; **dropped before modeling** so the model does not “memorise” IDs       |
| `Delivery_Time_min`      | `float`     | Actual delivery time in minutes                          | **Target variable** we want to predict                                                          |
| `Distance_km`            | `float`     | Distance from restaurant to customer, in kilometres      | Numeric; check for outliers; may interact with traffic (`Dist_x_Traffic[...]`)                  |
| `Preparation_Time_min`   | `float`     | Time the kitchen spends preparing the order              | Numeric; may interact with weather (`Prep_x_Weather[...]`)                                      |
| `Courier_Experience_yrs` | `float`     | Years of experience the courier has                      | Coerced to numeric; typos like `Courier_Experince_yrs` are fixed and values imputed with median |
| `Weather`                | `category`  | Weather condition (e.g. *Clear, Rainy, Stormy*)          | Trimmed, cleaned, imputed with most frequent value; used for interactions with prep time        |
| `Traffic_Level`          | `category`  | Traffic condition (e.g. *Low, Medium, High*)             | Cleaned categories; also used to build flags and distance interactions                          |
| `Time_of_Day`            | `category`  | Rough time block (Night / Morning / Afternoon / Evening) | Cleaned, then turned into **cyclical features** (`tod_sin`, `tod_cos`) and an ordered code      |
| `Vehicle_Type`           | `category`  | Courier’s vehicle (e.g. *Bike, Motorbike, Car*)          | Encoded as one-hot categories for the model                                                     |
| `High_Traffic_Flag`      | `int (0/1)` | Derived: is traffic *High* or *Very High*?               | Feature engineered from `Traffic_Level` to capture “high stress” conditions                     |
| `Speed_km_per_min`       | `float`     | Derived: `Distance_km / Delivery_Time_min`               | Post-hoc feature to understand realistic speeds; care taken with division by zero               |
| `Prep_per_km`            | `float`     | Derived: `Preparation_Time_min / Distance_km`            | Helps compare prep effort relative to distance                                                  |
| `Weather_Time`           | `category`  | Combined label: `Weather` + `Time_of_Day`                | Captures joint effects like “Rainy_Evening” vs “Clear_Morning”                                  |

**General quality checks & preprocessing rules**
Before any model tries to learn from the data, the app performs a series of sensible, leak-proof steps:
- Type fixing and cleaning
  - Strip extra spaces from text (" High " -> "High").
  - Convert text-like columns to string / category.
  - Convert numeric-like columns to proper numbers and handle bad values ("N/A", "unknown" -> NaN → imputed).
- Missing-value treatment
  - Numeric features -> median imputation (robust to outliers).
  - Categorical features -> most frequent category.
- ID and text columns
  - Kept for debugging only (like order_id).
  - Removed from the feature matrix to avoid the model learning meaningless codes.
- Train–test split
  - Split once (train_test_split, 80/20).
  - All preprocessing pipelines (SimpleImputer, OneHotEncoder, StandardScaler) are fitted only on the training set, then applied to test.
- Model-ready encoding
  - Numeric features: imputed, and sometimes scaled (for linear models like Ridge/Lasso).
  - Categorical features: one-hot encoded with handle_unknown="ignore" so new levels at prediction time do not break the app.
This ensures that anything you see in the app—charts, metrics, feature importance is based on clean, consistent data, and can be reproduced.

# Attachment
- [Data Processing](https://colab.research.google.com/drive/1vzjrYL14UJxsd2FptWBJTCzvRgBI1B0A)

# What We Bring To The Table? 

## Why the Model Is Useful (via “minutes within tolerance”)?

**What are we trying to answer here?**
- The app does not stop at MAE/RMSE, it explicitly measures how many orders have prediction errors within a chosen number of minutes.
- This _“minutes within tolerance”_ view answers a practical question: _“In reality, how often does our ETA land close enough to be acceptable?”_
- It turns the model from a purely statistical object into an operational signal of reliability.

**How do we do this in code?**
- Several model families (Ridge, Lasso, Decision Tree, Random Forest, and XGBoost when available) are trained and tuned using GridSearchCV with MAE as the main scoring metric, all under consistent preprocessing pipelines (preprocess_linear and preprocess_tree).
- The pipeline with the best cross validated MAE is chosen as best_pipe and then refitted with engineered features on X_train_fe, y_train (time-of-day encodings, trigonometric time features, traffic × distance and weather × prep interactions, speed, and prep-per-km).
- Using this final model (final_model), the app generates predictions for both train and test (y_tr_pred, y_te_pred) and summarises MAE, RMSE, and R² for each split in df_final_eval.
- On top of that, it computes for each test order whether |y_test − y_te_pred| falls within a user-selected tolerance window, producing a minutes-within-tolerance metric on unseen data.

**What do we actually get out of it?**
- A final evaluation table clearly reports Train vs Test MAE, RMSE, and R² after tuning and feature engineering, making it easy to see whether the model generalises beyond the training set.

<p align="center">
  <img src="https://github.com/Catherinerezi/Food-Delivery-ETA-Prediction/blob/main/aseets/Tabel%20perbandingan%20model.png" alt="Tabel Perbandingan" width="1000">
</p>

- From the same predictions, the minutes-within-tolerance metric is derived, providing a direct measure of how often ETA errors stay inside the chosen window on the test set.
- Together, these outputs show that the model is not only numerically strong, but also structured to support a reliability view that can be used later as the basis for ETA evaluation.

<p align="center">
  <img src="https://github.com/Catherinerezi/Food-Delivery-ETA-Prediction/blob/main/aseets/Visualisasi%20pemodelan%20terbaik.png" alt="Visualisasi Perbandingan" width="1000">
</p>

## How Big the Problem Is (the shape of delivery times & data quality)?

**What are we trying to understand about the data?**
- Before any modelling, the app answers: _“What does our data look like?”_ and _“How variable is Delivery_Time_min on its own?”_.
- It inspects **data quality, target distribution, and basic relationships with the target**, to understand how challenging ETA prediction will be.

**How do we explore this in practice?**
- The dataset is loaded from Google Drive via load_data(), copied into df, and profiled using:
  - **Shape & Head** (df.shape, df.head()) to show scale and schema.
  - **Describe** (df.describe().T) to capture basic statistics for numeric fields.
  - **Dtypes** (JSON mapping) to make each column’s type explicit.
- Data quality is assessed by:
  - Counting duplicates with df.duplicated().sum().
  - Computing missing percentages per column and visualising them with an Altair bar chart.
- A short cleaning pass:
  - Normalises categorical strings (strip + collapse whitespace).
  - Fills missing categoricals with their mode.
  - Coerces Courier_Experience_yrs to numeric and imputes its median, or stops with a clear error if the column is absent.
  - Recomputes missing and duplicate statistics after cleaning to show improvement.
- The dataset is split into train and test (train_test_split, 80/20), dropping ID-like columns and using Delivery_Time_min as the target.
- Light EDA then runs on the training data:
  - Missing percentage by feature (miss_train).
  - Target summary statistics from y_train.
  - A histogram of Delivery_Time_min on the cleaned data.
  - Correlations between numerical features and the target.
  - A boxplot of the target versus a selected low-cardinality categorical feature.

**What picture of the problem do we see?**
- A Data Understanding section that displays:
  - Dataset shape and the first rows, giving a concrete feel of what each record looks like.
  - Numeric summaries and data types, clarifying how each field is stored and used.
  - Duplicate and missing-value counts both before and after cleaning, showing how much preparation is needed.

<p align="center">
  <img src="https://github.com/Catherinerezi/Food-Delivery-ETA-Prediction/blob/main/aseets/Before%20Cleaning%20Data.png" alt="Before Data Cleaning" width="1000">
  Before Data Cleaning
</p>

<p align="center">
  <img src="https://github.com/Catherinerezi/Food-Delivery-ETA-Prediction/blob/main/aseets/After%20Data%20Cleaning.png" alt="Before Data Cleaning" width="1000">
  After Data Cleaning
</p>

- A histogram of Delivery_Time_min reveals:
  - Where most deliveries cluster in time.
  - How frequent very short or very long deliveries are.
  - Whether the target distribution is roughly tight or strongly skewed.
- Correlation and boxplot views highlight _"Which features seem most related to delivery time?"_ and _"How different categories shift the target, framing, and how difficult the prediction task is even before any model is trained?"_.
