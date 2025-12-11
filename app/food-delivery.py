# -*- coding: utf-8 -*-
"""
TakeHomeTestDS_Chatharina_Cheisha_streamlit.py

Streamlit app for:
- Food-delivery ETA prediction (Delivery_Time_min)
- Exploratory data analysis
- Model training & tuning (Ridge/Lasso/Tree/RF/XGBoost*)
- Feature importance (intrinsic + permutation)
- Diagnostics & PDP
"""

import os
import datetime as dt
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    make_scorer,
)
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# Optional XGBoost
has_xgb = False
try:
    from xgboost import XGBRegressor
    has_xgb = True
except Exception:
    has_xgb = False

st.set_page_config(
    page_title="Prediksi Waktu Pengantaran Makanan",
    layout="wide",
)

st.title("📦 Prediksi Waktu Pengantaran Makanan (ETA)")
st.caption(
    "Take-home test — Food-delivery ETA prediction with sklearn pipelines, tuning, "
    "Altair interactive visualizations, and feature engineering."
)

with st.expander("🎯 Tujuan", expanded=True):
    st.markdown(
        """
**Goal**  
Membangun model prediktif untuk memperkirakan `Delivery_Time_min` (menit) per pesanan
pada layanan pengantaran makanan, sehingga tim operasional dapat:

- Mengestimasi ETA yang lebih akurat  
- Mengoptimalkan alokasi kurir  
- Memahami faktor-faktor yang paling mempengaruhi waktu pengantaran
"""
    )

# Data Loading 
@st.cache_data(show_spinner=True)
def load_data():
    file_id = "1qI18G7Rjr5Axqz-HawadpTzSnTwf5jeQ"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    df = pd.read_csv(url)
    return df


df_raw = load_data()
df = df_raw.copy()

# Data Undertanding
st.header("Data Undertanding")

c1, c2, c3 = st.columns([2, 2, 3])

with c1:
    st.subheader("Shape & Head")
    st.write("Shape:", df.shape)
    st.dataframe(df.head(), use_container_width=True)

with c2:
    st.subheader("Describe")
    st.dataframe(df.describe().T, use_container_width=True)

with c3:
    st.subheader("Dtypes")
    st.json({col: str(tp) for col, tp in df.dtypes.items()})


# Cek Duplikat
st.subheader("Cek Duplikat")
dup_count = int(df.duplicated().sum())
st.write(f"Jumlah duplikat: **{dup_count}**")

# Cek Missing Value
st.subheader("Cek Missing Value")

missing_pct = (df.isna().mean() * 100).sort_values(ascending=False).round(2)
missing_df = missing_pct.reset_index()
missing_df.columns = ["column", "missing_pct"]

c_m1, c_m2 = st.columns(2)
with c_m1:
    st.dataframe(missing_df, use_container_width=True)

with c_m2:
    st.caption("Missing value (%) per column")
    chart_missing = (
        alt.Chart(missing_df)
        .mark_bar()
        .encode(
            x=alt.X("missing_pct:Q", title="% Missing"),
            y=alt.Y("column:N", sort="-x", title="Column"),
            tooltip=["column", alt.Tooltip("missing_pct:Q", format=".2f")],
        )
        .properties(height=max(200, 20 * len(missing_df)))
        .interactive()
    )
    st.altair_chart(chart_missing, use_container_width=True)

# Data Cleaning (tambahan)
st.subheader("Cleaning singkat (kategori & numerik)")

cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
for c in cat_cols:
    df[c] = (
        df[c]
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

for c in cat_cols:
    m = df[c].mode(dropna=True)
    if not m.empty:
        df[c] = df[c].fillna(m.iat[0])

candidate_cols = ["Courier_Experience_yrs"]
col_ce = next((c for c in candidate_cols if c in df.columns), None)
if col_ce is None:
    st.error(
        f"Kolom pengalaman kurir tidak ditemukan. Kolom tersedia: {list(df.columns)}"
    )
    st.stop()

df[col_ce] = pd.to_numeric(df[col_ce], errors="coerce")
df[col_ce] = df[col_ce].fillna(df[col_ce].median(skipna=True))

missing_pct_after = (df.isna().mean() * 100).sort_values(ascending=False).round(2)
dup_count_after = int(df.duplicated().sum())

c_m3, c_m4 = st.columns(2)
with c_m3:
    st.write("Missing value setelah cleaning:")
    st.dataframe(
        missing_pct_after.reset_index().rename(columns={"index": "column", 0: "missing_pct"}),
        use_container_width=True,
    )
with c_m4:
    st.write(f"Duplikat setelah cleaning: **{dup_count_after}**")

# Pembagian Data Set
st.header("Pembagian Data Set")

TARGET = "Delivery_Time_min"
assert TARGET in df.columns, f"Target '{TARGET}' tidak ada di kolom."

id_like = [c for c in df.columns if c.lower() in {"order_id", "id"}]
X = df.drop(columns=[TARGET] + id_like, errors="ignore")
y = df[TARGET]

# handle typo
if "Courier_Experince_yrs" in X.columns and "Courier_Experience_yrs" not in X.columns:
    X = X.rename(columns={"Courier_Experince_yrs": "Courier_Experience_yrs"})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
st.success(f"X_train: {X_train.shape} | X_test: {X_test.shape}")


# EDA ringan + feature importance eksploratif
st.header("EDA ringan + feature importance eksploratif")

# EDA ringan di train
st.subheader("EDA ringan di train")

num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()

c_eda1, c_eda2 = st.columns(2)
with c_eda1:
    miss_train = (X_train.isna().mean() * 100).sort_values(ascending=False).round(2)
    st.write("Missing % per kolom (TRAIN)")
    st.dataframe(miss_train.rename("%missing").to_frame(), use_container_width=True)

with c_eda2:
    st.write("Target stats (TRAIN)")
    st.dataframe(
        y_train.describe()[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
        .round(2)
        .to_frame()
        .T,
        use_container_width=True,
    )

# Distribusi target
st.subheader("Distribusi Target: Delivery_Time_min")
hist_target = (
    alt.Chart(df)
    .mark_bar()
    .encode(
        x=alt.X("Delivery_Time_min:Q", bin=alt.Bin(maxbins=30), title="Delivery_Time_min"),
        y=alt.Y("count():Q", title="Count"),
        tooltip=[alt.Tooltip("count():Q", title="Count")],
    )
    .properties(height=300)
    .interactive()
)
st.altair_chart(hist_target, use_container_width=True)

# Korelasi numerik vs target
if len(num_cols) > 1:
    corr = df[num_cols + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(
        ascending=False
    )
    corr_df = corr.round(3).reset_index().rename(
        columns={"index": "feature", TARGET: "corr"}
    )
    st.subheader("Korelasi numerik vs target")
    corr_chart = (
        alt.Chart(corr_df)
        .mark_bar()
        .encode(
            x=alt.X("corr:Q", title="Correlation with target"),
            y=alt.Y("feature:N", sort="-x"),
            tooltip=["feature", alt.Tooltip("corr:Q", format=".3f")],
        )
        .properties(height=max(200, 20 * len(corr_df)))
        .interactive()
    )
    st.altair_chart(corr_chart, use_container_width=True)

# Analisis kategori vs target
st.subheader("Delivery_Time_min vs Kategori (Boxplot)")

cat_cols_for_box = [
    c for c in df.select_dtypes(include=["object", "string"]).columns if df[c].nunique() <= 10
]
if cat_cols_for_box:
    col_sel = st.selectbox("Pilih kolom kategori:", cat_cols_for_box, index=0)
    box_df = df[[col_sel, TARGET]].dropna()
    box_chart = (
        alt.Chart(box_df)
        .mark_boxplot()
        .encode(
            x=alt.X(f"{col_sel}:N", title=col_sel),
            y=alt.Y(f"{TARGET}:Q", title="Delivery_Time_min"),
            tooltip=[col_sel, alt.Tooltip(TARGET, format=".2f")],
        )
        .properties(height=350)
        .interactive()
    )
    st.altair_chart(box_chart, use_container_width=True)
else:
    st.info("Tidak ada kolom kategori dengan cardinality ≤ 10.")

# Feature Imprtance Eksploratif (Permutation importance via CV)
st.subheader("Feature Imprtance Eksploratif (ΔMAE via Permutation)")

# Preprocess untuk importance
num_transform = Pipeline([("imputer", SimpleImputer(strategy="median"))])
cat_transform = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", num_transform, num_cols),
        ("cat", cat_transform, cat_cols),
    ],
    remainder="drop",
)

rf_imp = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
pipe_rf = Pipeline([("preprocess", preprocess), ("model", rf_imp)])

cv = KFold(n_splits=5, shuffle=True, random_state=42)
feat_orig = list(num_cols) + list(cat_cols)
imp_mae_folds = []

for k, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train), 1):
    X_tr, X_va = X_train.iloc[tr_idx].copy(), X_train.iloc[va_idx].copy()
    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    pipe_rf.fit(X_tr, y_tr)
    yhat_base = pipe_rf.predict(X_va)
    base_mae = mean_absolute_error(y_va, yhat_base)

    rng = np.random.default_rng(42 + k)
    d_mae = {}
    for f in feat_orig:
        Xp = X_va.copy()
        v = Xp[f].to_numpy(copy=True)
        rng.shuffle(v)
        Xp[f] = v
        yhat_p = pipe_rf.predict(Xp)
        d_mae[f] = mean_absolute_error(y_va, yhat_p) - base_mae
    imp_mae_folds.append(d_mae)


def _avg(tbls):
    keys = set().union(*[t.keys() for t in tbls])
    return (
        pd.Series({k: np.mean([t.get(k, 0.0) for t in tbls]) for k in keys})
        .sort_values(ascending=False)
        .reset_index()
    )


imp_mae_tbl = _avg(imp_mae_folds)
imp_mae_tbl.columns = ["feature", "perm_importance_MAE"]

chart_imp = (
    alt.Chart(imp_mae_tbl.head(20))
    .mark_bar()
    .encode(
        x=alt.X("perm_importance_MAE:Q", title="ΔMAE (lebih besar = lebih penting)"),
        y=alt.Y("feature:N", sort="-x"),
        tooltip=[
            "feature",
            alt.Tooltip("perm_importance_MAE:Q", format=".4f", title="ΔMAE"),
        ],
    )
    .properties(height=400)
    .interactive()
)
st.altair_chart(chart_imp, use_container_width=True)


# Tuning via CV (MAE) + Preprocessing
st.header("Tuning via CV")

def rmse_metric(y_true, y_pred):
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return np.sqrt(mean_squared_error(y_true, y_pred))


if "num_cols" not in globals() or "cat_cols" not in globals():
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

preprocess_tree = ColumnTransformer(
    [
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
        (
            "cat",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            cat_cols,
        ),
    ]
)

preprocess_linear = ColumnTransformer(
    [
        (
            "num",
            Pipeline(
                [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
            ),
            num_cols,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            cat_cols,
        ),
    ]
)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = "neg_mean_absolute_error"

cands = [
    (
        "Ridge",
        Pipeline([("preprocess", preprocess_linear), ("model", Ridge())]),
        {"model__alpha": [0.1, 1.0, 10.0]},
    ),
    (
        "Lasso",
        Pipeline(
            [("preprocess", preprocess_linear), ("model", Lasso(max_iter=10000))]
        ),
        {"model__alpha": [0.001, 0.01, 0.1, 1.0]},
    ),
    (
        "Decision Tree",
        Pipeline(
            [
                ("preprocess", preprocess_tree),
                ("model", DecisionTreeRegressor(random_state=42)),
            ]
        ),
        {"model__max_depth": [3, 5, 8, None], "model__min_samples_leaf": [1, 3, 5]},
    ),
    (
        "Random Forest",
        Pipeline(
            [
                ("preprocess", preprocess_tree),
                (
                    "model",
                    RandomForestRegressor(random_state=42, n_jobs=-1),
                ),
            ]
        ),
        {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 8, 12],
            "model__min_samples_leaf": [1, 2, 4],
        },
    ),
]

if has_xgb:
    cands.append(
        (
            "XGBoost",
            Pipeline(
                [
                    ("preprocess", preprocess_tree),
                    (
                        "model",
                        XGBRegressor(
                            random_state=42,
                            n_jobs=-1,
                            tree_method="hist",
                            eval_metric="mae",
                        ),
                    ),
                ]
            ),
            {
                "model__n_estimators": [300, 600],
                "model__max_depth": [3, 6],
                "model__learning_rate": [0.05, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
            },
        )
    )

rows, best_models = [], {}
for name, pipe_cand, grid in cands:
    pipe_cand.fit(X_train, y_train)
    y_hat_pre = pipe_cand.predict(X_test)
    mae_pre = mean_absolute_error(y_test, y_hat_pre)
    rmse_pre = rmse_metric(y_test, y_hat_pre)
    r2_pre = r2_score(y_test, y_hat_pre)

    gs = GridSearchCV(
        pipe_cand,
        grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    gs.fit(X_train, y_train)
    best_models[name] = gs.best_estimator_

    y_hat_post = gs.predict(X_test)
    rows.append(
        {
            "model": name,
            "mae_pre": mae_pre,
            "rmse_pre": rmse_pre,
            "r2_pre": r2_pre,
            "mae_post": mean_absolute_error(y_test, y_hat_post),
            "rmse_post": rmse_metric(y_test, y_hat_post),
            "r2_post": r2_score(y_test, y_hat_post),
            "best_params": gs.best_params_,
        }
    )

cmp = pd.DataFrame(rows).sort_values("mae_post").reset_index(drop=True)
st.subheader("Perbandingan Model (Before vs After Tuning)")
st.dataframe(cmp, use_container_width=True)

best_name = cmp.loc[0, "model"]
best_pipe = best_models[best_name]
st.success(f"Best by CV (MAE): **{best_name}**")


# Pemilihan Model & Evaluasi baseline
st.header("Pemilihan Model & Evaluasi (Ridge / Lasso / Tree / RF / XGB)")

def init_supported(estimator_cls, **kwargs):
    from inspect import signature

    params = signature(estimator_cls.__init__).parameters
    filtered = {k: v for k, v in kwargs.items() if k in params}
    return estimator_cls(**filtered)


def evaluate_model(model, X_tr, y_tr, X_te, y_te, name="Model"):
    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)
    mae_tr = mean_absolute_error(y_tr, pred_tr)
    mae_te = mean_absolute_error(y_te, pred_te)
    rmse_te = np.sqrt(mean_squared_error(y_te, pred_te))
    r2_te = r2_score(y_te, pred_te)
    return {"name": name, "mae": mae_te, "rmse": rmse_te, "r2": r2_te}


results = []

ridge_pipe = Pipeline(
    [
        ("preprocess", preprocess_linear),
        ("model", init_supported(Ridge, alpha=1.0, random_state=42)),
    ]
)
ridge_pipe.fit(X_train, y_train)
results.append(evaluate_model(ridge_pipe, X_train, y_train, X_test, y_test, "Ridge"))

lasso_pipe = Pipeline(
    [
        ("preprocess", preprocess_linear),
        ("model", init_supported(Lasso, alpha=0.001, random_state=42, max_iter=10000)),
    ]
)
lasso_pipe.fit(X_train, y_train)
results.append(evaluate_model(lasso_pipe, X_train, y_train, X_test, y_test, "Lasso"))

tree_pipe = Pipeline(
    [
        ("preprocess", preprocess_tree),
        ("model", init_supported(DecisionTreeRegressor, random_state=42, max_depth=None)),
    ]
)
tree_pipe.fit(X_train, y_train)
results.append(
    evaluate_model(tree_pipe, X_train, y_train, X_test, y_test, "Decision Tree")
)

rf_pipe = Pipeline(
    [
        ("preprocess", preprocess_tree),
        (
            "model",
            init_supported(
                RandomForestRegressor,
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)
rf_pipe.fit(X_train, y_train)
results.append(
    evaluate_model(rf_pipe, X_train, y_train, X_test, y_test, "Random Forest")
)

if has_xgb:
    xgb_pipe = Pipeline(
        [
            ("preprocess", preprocess_tree),
            (
                "model",
                init_supported(
                    XGBRegressor,
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=-1,
                    tree_method="hist",
                ),
            ),
        ]
    )
    xgb_pipe.fit(X_train, y_train)
    results.append(
        evaluate_model(xgb_pipe, X_train, y_train, X_test, y_test, "XGBoost")
    )

df_res = pd.DataFrame(results, columns=["name", "mae", "rmse", "r2"])
df_res = df_res.sort_values("mae").reset_index(drop=True)
df_res.insert(0, "rank", range(1, len(df_res) + 1))
df_res[["mae", "rmse", "r2"]] = df_res[["mae", "rmse", "r2"]].round(3)

st.subheader("Tabel Ringkasan (Evaluasi)")
st.dataframe(df_res, use_container_width=True)

# Visualisasi RMSE / MAE / R² (Altair)
st.subheader("Visualisasi RMSE / MAE / R²")

# Baseline (mean)
yhat_base = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)
baseline_rmse = np.sqrt(np.mean((y_test - yhat_base) ** 2))
baseline_mae = np.mean(np.abs(y_test - yhat_base))
baseline_r2 = 1 - np.sum((y_test - yhat_base) ** 2) / np.sum(
    (y_test - y_test.mean()) ** 2
)

# RMSE
df_rmse = pd.concat(
    [
        pd.DataFrame([{"name": "Baseline (Mean)", "rmse": baseline_rmse}]),
        df_res[["name", "rmse"]].rename(columns={"rmse": "rmse"}),
    ],
    ignore_index=True,
)
df_rmse["rmse"] = df_rmse["rmse"].round(3)
chart_rmse = (
    alt.Chart(df_rmse)
    .mark_bar()
    .encode(
        x=alt.X("rmse:Q", title="RMSE (lebih kecil = lebih baik)"),
        y=alt.Y("name:N", sort="x", title="Model"),
        tooltip=["name", alt.Tooltip("rmse:Q", format=".3f")],
    )
    .properties(height=max(200, 25 * len(df_rmse)), title="Perbandingan RMSE")
    .interactive()
)

# MAE
df_mae = pd.concat(
    [
        pd.DataFrame([{"name": "Baseline (Mean)", "mae": baseline_mae}]),
        df_res[["name", "mae"]].rename(columns={"mae": "mae"}),
    ],
    ignore_index=True,
)
df_mae["mae"] = df_mae["mae"].round(3)
chart_mae = (
    alt.Chart(df_mae)
    .mark_bar()
    .encode(
        x=alt.X("mae:Q", title="MAE (lebih kecil = lebih baik)"),
        y=alt.Y("name:N", sort="x", title="Model"),
        tooltip=["name", alt.Tooltip("mae:Q", format=".3f")],
    )
    .properties(height=max(200, 25 * len(df_mae)), title="Perbandingan MAE")
    .interactive()
)

# R²
df_r2 = pd.concat(
    [
        pd.DataFrame([{"name": "Baseline (Mean)", "r2": baseline_r2}]),
        df_res[["name", "r2"]].rename(columns={"r2": "r2"}),
    ],
    ignore_index=True,
)
df_r2["r2"] = df_r2["r2"].round(3)
chart_r2 = (
    alt.Chart(df_r2)
    .mark_bar()
    .encode(
        x=alt.X("r2:Q", title="R² (lebih besar = lebih baik)"),
        y=alt.Y("name:N", sort="-x", title="Model"),
        tooltip=["name", alt.Tooltip("r2:Q", format=".3f")],
    )
    .properties(height=max(200, 25 * len(df_r2)), title="Perbandingan R²")
    .interactive()
)

c_v1, c_v2, c_v3 = st.columns(3)
with c_v1:
    st.altair_chart(chart_rmse, use_container_width=True)
with c_v2:
    st.altair_chart(chart_mae, use_container_width=True)
with c_v3:
    st.altair_chart(chart_r2, use_container_width=True)

# Feature Importance untuk model terbaik
st.header("Feature Importance untuk model terbaik")

# mapping nama
model_map = {
    "Ridge": ridge_pipe,
    "Lasso": lasso_pipe,
    "Decision Tree": tree_pipe,
    "Random Forest": rf_pipe,
    "XGBoost": xgb_pipe if has_xgb else None,
}
pipe = model_map.get(best_name, best_pipe)
if pipe is None:
    pipe = best_pipe

ct = pipe.named_steps["preprocess"]
try:
    feat_names_pre = ct.get_feature_names_out()
    feat_names_pre = list(feat_names_pre)
except Exception:
    num_names = list(ct.transformers_[0][2])
    cat_base = ct.transformers_[1][2]
    ohe = ct.named_transformers_["cat"].named_steps["ohe"]
    feat_names_pre = num_names + list(ohe.get_feature_names_out(cat_base))

mdl = pipe.named_steps["model"]

# Intrinsic importance
st.subheader("Intrinsic Importance / Coefficients")

fi_intrinsic = None
if hasattr(mdl, "feature_importances_"):
    fi_intrinsic = pd.DataFrame(
        {"feature": feat_names_pre, "importance": mdl.feature_importances_}
    ).sort_values("importance", ascending=False)
elif hasattr(mdl, "coef_"):
    coef = np.ravel(mdl.coef_)
    fi_intrinsic = pd.DataFrame(
        {
            "feature": feat_names_pre,
            "importance": np.abs(coef),
        }
    ).sort_values("importance", ascending=False)

if fi_intrinsic is not None:
    chart_int = (
        alt.Chart(fi_intrinsic.head(20))
        .mark_bar()
        .encode(
            x=alt.X("importance:Q", title="Importance / |Coefficient|"),
            y=alt.Y("feature:N", sort="-x"),
            tooltip=[
                "feature",
                alt.Tooltip("importance:Q", format=".4f", title="Importance"),
            ],
        )
        .properties(height=400)
        .interactive()
    )
    st.altair_chart(chart_int, use_container_width=True)
else:
    st.info(f"Model {best_name} tidak expose intrinsic importance/coef.")

# Permutation importance (model-agnostic)
st.subheader("Permutation importance (model-agnostic, ΔMAE)")

r_perm = permutation_importance(
    pipe,
    X_test,
    y_test,
    n_repeats=20,
    random_state=42,
    n_jobs=-1,
    scoring=lambda est, X, y: -mean_absolute_error(y, est.predict(X)),
)
imp_perm = -r_perm.importances_mean

feat_names_perm = np.array(feat_names_pre)
k_min = min(len(feat_names_perm), len(imp_perm))

perm_mae = (
    pd.DataFrame(
        {
            "feature": feat_names_perm[:k_min],
            "perm_importance_MAE": imp_perm[:k_min],
        }
    )
    .sort_values("perm_importance_MAE", ascending=False)
    .reset_index(drop=True)
)

chart_perm = (
    alt.Chart(perm_mae.head(20))
    .mark_bar()
    .encode(
        x=alt.X("perm_importance_MAE:Q", title="ΔMAE"),
        y=alt.Y("feature:N", sort="-x"),
        tooltip=[
            "feature",
            alt.Tooltip("perm_importance_MAE:Q", format=".4f", title="ΔMAE"),
        ],
    )
    .properties(height=400)
    .interactive()
)
st.altair_chart(chart_perm, use_container_width=True)

# Visualisasi — Parity + Residuals + Error per segmen
st.header("Visualisasi")

st.subheader("Parity + Residuals")
y_pred_pipe = pipe.predict(X_test)
resid = y_test - y_pred_pipe
diag = pd.DataFrame(
    {"actual": y_test, "pred": y_pred_pipe, "resid": resid}
).reset_index(drop=True)

c_diag1, c_diag2, c_diag3 = st.columns(3)

with c_diag1:
    st.caption("Parity Plot — Actual vs Predicted")
    parity_chart = (
        alt.Chart(diag)
        .mark_circle(opacity=0.6)
        .encode(
            x=alt.X("actual:Q", title="Actual"),
            y=alt.Y("pred:Q", title="Predicted"),
            tooltip=[
                alt.Tooltip("actual:Q", format=".2f"),
                alt.Tooltip("pred:Q", format=".2f"),
                alt.Tooltip("resid:Q", format=".2f"),
            ],
        )
        .interactive()
    )
    st.altair_chart(parity_chart, use_container_width=True)

with c_diag2:
    st.caption("Residual Histogram")
    resid_hist = (
        alt.Chart(diag)
        .mark_bar()
        .encode(
            x=alt.X("resid:Q", bin=alt.Bin(maxbins=30), title="Residual"),
            y=alt.Y("count():Q", title="Count"),
            tooltip=[alt.Tooltip("count():Q", title="Count")],
        )
        .interactive()
    )
    st.altair_chart(resid_hist, use_container_width=True)

with c_diag3:
    st.caption("Residual vs Predicted")
    rule_zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule().encode(y="y:Q")
    resid_vs_pred = (
        alt.Chart(diag)
        .mark_circle(opacity=0.6)
        .encode(
            x=alt.X("pred:Q", title="Predicted"),
            y=alt.Y("resid:Q", title="Residual"),
            tooltip=[
                alt.Tooltip("pred:Q", format=".2f"),
                alt.Tooltip("resid:Q", format=".2f"),
            ],
        )
        .interactive()
    )
    st.altair_chart(resid_vs_pred + rule_zero, use_container_width=True)

# Error per segmen
st.subheader("Error per segmen")

df_te = X_test.copy()
df_te["y_true"] = y_test
df_te["y_pred"] = y_pred_pipe
df_te["abs_err"] = (df_te["y_true"] - df_te["y_pred"]).abs()

seg_cols = [c for c in ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"] if c in df_te.columns]

if seg_cols:
    seg_col = st.selectbox("Pilih kolom segmentasi:", seg_cols, index=0)
    seg = (
        df_te.groupby(seg_col)["abs_err"]
        .agg(count="count", mean="mean", median="median", max="max")
        .reset_index()
        .sort_values("mean")
    )
    st.write(f"Segment MAE by **{seg_col}**")
    st.dataframe(seg, use_container_width=True)

    # Bar chart mean abs_err
    seg_chart = (
        alt.Chart(seg)
        .mark_bar()
        .encode(
            x=alt.X("mean:Q", title="Mean |abs_err| (min)"),
            y=alt.Y(f"{seg_col}:N", sort="-x", title=seg_col),
            tooltip=[
                seg_col,
                alt.Tooltip("mean:Q", format=".2f", title="Mean |abs_err|"),
                alt.Tooltip("median:Q", format=".2f", title="Median |abs_err|"),
                "count:Q",
                alt.Tooltip("max:Q", format=".2f", title="Max |abs_err|"),
            ],
        )
        .properties(height=max(200, 25 * len(seg)))
        .interactive()
    )
    st.altair_chart(seg_chart, use_container_width=True)
else:
    st.info("Tidak ada kolom kategori untuk error per segmen.")

# PDP (Partial Dependence) untuk fitur numerik utama
st.header("PDP (Partial Dependence) untuk fitur numerik utama")

def compute_pdp_1d(model, X_ref: pd.DataFrame, feat: str, grid_size: int = 25):
    vals = np.linspace(
        X_ref[feat].quantile(0.05),
        X_ref[feat].quantile(0.95),
        grid_size,
    )
    pdp_vals = []
    X_temp = X_ref.copy()
    for v in vals:
        X_temp[feat] = v
        pdp_vals.append(model.predict(X_temp).mean())
    return pd.DataFrame({feat: vals, "pred": pdp_vals})


pdp_candidates = [
    f for f in ["Distance_km", "Preparation_Time_min", "Courier_Experience_yrs"] if f in X_test.columns
]
if not pdp_candidates:
    pdp_candidates = num_cols[: min(3, len(num_cols))]

if pdp_candidates:
    feat_pdp = st.selectbox("Pilih fitur numerik untuk PDP 1D:", pdp_candidates, index=0)
    pdp_df = compute_pdp_1d(pipe, X_test.copy(), feat_pdp, grid_size=40)
    chart_pdp = (
        alt.Chart(pdp_df)
        .mark_line()
        .encode(
            x=alt.X(f"{feat_pdp}:Q", title=feat_pdp),
            y=alt.Y("pred:Q", title="Predicted Delivery_Time_min"),
            tooltip=[alt.Tooltip(feat_pdp, format=".2f"), alt.Tooltip("pred:Q", format=".2f")],
        )
        .properties(height=350, title=f"PDP — {feat_pdp}")
        .interactive()
    )
    st.altair_chart(chart_pdp, use_container_width=True)
else:
    st.info("Tidak ada fitur numerik untuk PDP.")

# Feature engineering (Time_of_Day, interaksi, dsb.)
st.header("Feature engineering")

def add_time_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df_out = df_in.copy()
    if "Time_of_Day" in df_out.columns:
        order = {"Night": 0, "Morning": 1, "Afternoon": 2, "Evening": 3}
        df_out["Time_of_Day_ord"] = df_out["Time_of_Day"].map(order)

        hour_map = {"Night": 2, "Morning": 9, "Afternoon": 15, "Evening": 20}
        h = df_out["Time_of_Day"].map(hour_map).astype(float)
        h = h.fillna(np.nanmedian(h))
        df_out["tod_sin"] = np.sin(2 * np.pi * h / 24.0)
        df_out["tod_cos"] = np.cos(2 * np.pi * h / 24.0)
    return df_out


def build_interaction_maker(X_train_src: pd.DataFrame):
    levels_traffic = (
        X_train_src["Traffic_Level"].dropna().unique().tolist()
        if "Traffic_Level" in X_train_src.columns
        else []
    )
    levels_weather = (
        X_train_src["Weather"].dropna().unique().tolist()
        if "Weather" in X_train_src.columns
        else []
    )

    def add_interactions(df_in: pd.DataFrame) -> pd.DataFrame:
        df_out = df_in.copy()
        if "Distance_km" in df_out.columns and levels_traffic:
            for lv in levels_traffic:
                col = f"Dist_x_Traffic[{lv}]"
                df_out[col] = np.where(
                    df_out.get("Traffic_Level") == lv, df_out["Distance_km"], 0.0
                )
        if "Preparation_Time_min" in df_out.columns and levels_weather:
            for lv in levels_weather:
                col = f"Prep_x_Weather[{lv}]"
                df_out[col] = np.where(
                    df_out.get("Weather") == lv,
                    df_out["Preparation_Time_min"],
                    0.0,
                )
        return df_out

    return add_interactions


def add_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df_out = df_in.copy()
    if all(c in df_out.columns for c in ["Distance_km", "Delivery_Time_min"]):
        df_out["Speed_km_per_min"] = df_out["Distance_km"] / df_out["Delivery_Time_min"].replace(
            0, np.nan
        )
    if all(c in df_out.columns for c in ["Preparation_Time_min", "Distance_km"]):
        df_out["Prep_per_km"] = df_out["Preparation_Time_min"] / df_out["Distance_km"].replace(
            0, np.nan
        )
    if all(c in df_out.columns for c in ["Weather", "Time_of_Day"]):
        df_out["Weather_Time"] = (
            df_out["Weather"].astype(str) + "_" + df_out["Time_of_Day"].astype(str)
        )
    if "Traffic_Level" in df_out.columns:
        df_out["High_Traffic_Flag"] = df_out["Traffic_Level"].isin(
            ["High", "Very High"]
        ).astype(int)
    return df_out


X_train_fe = add_time_features(X_train)
X_test_fe = add_time_features(X_test)
add_interactions = build_interaction_maker(X_train_fe)
X_train_fe = add_interactions(X_train_fe)
X_test_fe = add_interactions(X_test_fe)
X_train_fe = add_features(X_train_fe)
X_test_fe = add_features(X_test_fe)

num_cols_fe = X_train_fe.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_fe = X_train_fe.select_dtypes(include=["object", "string"]).columns.tolist()

st.write(
    "Contoh fitur numerik baru:",
    [c for c in ["Time_of_Day_ord", "tod_sin", "tod_cos", "Prep_per_km"] if c in num_cols_fe],
)
st.write(
    "Contoh fitur interaksi:",
    [c for c in X_train_fe.columns if c.startswith(("Dist_x_Traffic", "Prep_x_Weather"))][:6],
)

# Refit best_pipe dengan FE
try:
    best_pipe.fit(X_train_fe, y_train)
    y_pred_fe = best_pipe.predict(X_test_fe)
    mae_fe = mean_absolute_error(y_test, y_pred_fe)
    st.success(f"MAE dengan feature engineering (FE): **{mae_fe:.3f}**")
except Exception as e:
    st.warning(f"Refit dengan FE gagal, pakai fitur lama saja. Error: {e}")

# Final evaluation
st.header("Final evaluation (setelah tuning)")

final_model = best_pipe

y_tr_pred = final_model.predict(X_train_fe)
y_te_pred = final_model.predict(X_test_fe)

metrics_final = {
    "split": ["Train", "Test"],
    "MAE": [
        mean_absolute_error(y_train, y_tr_pred),
        mean_absolute_error(y_test, y_te_pred),
    ],
    "RMSE": [
        rmse_metric(y_train, y_tr_pred),
        rmse_metric(y_test, y_te_pred),
    ],
    "R2": [r2_score(y_train, y_tr_pred), r2_score(y_test, y_te_pred)],
}
df_final_eval = pd.DataFrame(metrics_final).round(3)
st.dataframe(df_final_eval, use_container_width=True)

st.subheader("Finalisasi — % within tolerance (donut)")

TOL_MIN = st.slider(
    "Toleransi menit (untuk % within)", min_value=1, max_value=15, value=5, step=1
)

pct_within = (np.abs(y_test - y_te_pred) <= TOL_MIN).mean() * 100.0
donut_df = pd.DataFrame(
    {"label": ["Within", "Outside"], "value": [pct_within, 100.0 - pct_within]}
)

donut_chart = (
    alt.Chart(donut_df)
    .mark_arc(innerRadius=70)
    .encode(
        theta="value:Q",
        color="label:N",
        tooltip=[
            "label:N",
            alt.Tooltip("value:Q", format=".1f", title="Percentage"),
        ],
    )
    .properties(
        width=300,
        height=300,
        title=f"% within ±{TOL_MIN} menit",
    )
    .interactive()
)
st.altair_chart(donut_chart, use_container_width=False)
st.write(f"**% within ±{TOL_MIN} menit:** {pct_within:.1f}%")

# Folder Artifacts (opsional di Streamlit)
st.header("Folder Artifacts (opsional)")

ART = None
for k in list(globals().keys()):
    if isinstance(globals()[k], str) and k.startswith("ART") and os.path.isdir(
        globals()[k]
    ):
        ART = globals()[k]
        break

if ART is None:
    ART = f"artifacts_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    try:
        os.makedirs(ART, exist_ok=True)
        st.info(f"Folder artifacts dibuat: `{ART}` (di environment server).")
    except Exception as e:
        st.warning(f"Gagal membuat folder artifacts (boleh diabaikan untuk Streamlit): {e}")
else:
    st.info(f"Menggunakan folder artifacts: `{ART}`")

st.markdown(
    """
> **Catatan**: Di environment Streamlit, folder ini tidak selalu mudah diakses user,
> tapi bagian *garis besar* `Folder Artifacts` tetap dipertahankan sesuai notebook.
"""
)
