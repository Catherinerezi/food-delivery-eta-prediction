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
