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

