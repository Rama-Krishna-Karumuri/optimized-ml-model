import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import shap
import warnings

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')


# ============================================================
# CONSTANTS
# ============================================================

TARGET = "Strength"

# Raw input columns (now includes Water)
BASE_COLS = ["Cement", "Water", "CoarseAgg", "FineAgg", "FlyAsh", "SilicaFume", "GGBFS", "SP"]

# Pre-computed features present in the dataset
PRECOMPUTED_COLS = ["WB_Ratio", "Total_Binder", "Agg_Binder_Ratio", "SF_Ratio"]

# Train file column names  (already named properly)
TRAIN_FILE = "dataset_80_with_features.xlsx"

# Test file has different column names — map them here
TEST_COL_MAP = {
    " Cement(kg/m3)":                       "Cement",
    " Water(kg/m3)":                        "Water",
    "Coarse aggregate(kg/m3)":              "CoarseAgg",
    "Fine aggregate(kg/m3)":               "FineAgg",
    " FA (kg/m3)":                          "FlyAsh",
    "SF (kg/m3)":                           "SilicaFume",
    "GGBFS (kg/m3)":                        "GGBFS",
    "SP (kg/m3)":                           "SP",
    "Cylinder compressive strength (MPa)":  "Strength",
}
TEST_FILE = "dataset_20_with_features.xlsx"

FEATURE_DISPLAY = {
    "Cement":     "Cement",
    "Water":      "Water",
    "CoarseAgg":  "Coarse Aggregate",
    "FineAgg":    "Fine Aggregate",
    "FlyAsh":     "Fly Ash",
    "SilicaFume": "Silica Fume",
    "GGBFS":      "GGBFS",
    "SP":         "Superplasticizer",
}


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive domain-aware feature engineering for concrete strength.
    Incorporates Water column and all pre-computed features.
    """
    d = df.copy()

    cement = d["Cement"].astype(float)
    water  = d["Water"].astype(float)
    coarse = d["CoarseAgg"].astype(float)
    fine   = d["FineAgg"].astype(float)
    fa     = d["FlyAsh"].astype(float)
    sf     = d["SilicaFume"].astype(float)
    ggbfs  = d["GGBFS"].astype(float)
    sp     = d["SP"].astype(float)

    total_binder = cement + fa + sf + ggbfs
    total_agg    = fine + coarse
    total_vol    = total_binder + total_agg + water + sp

    # ── Pre-computed features (already in dataset, re-derive for robustness) ──
    d["WB_Ratio"]         = water / (total_binder + 1e-6)
    d["Total_Binder"]     = total_binder
    d["Agg_Binder_Ratio"] = total_agg / (total_binder + 1e-6)
    d["SF_Ratio"]         = sf / (total_binder + 1e-6)

    # ── Water-cement ratio (key concrete design parameter) ────────────────────
    d["wc_ratio"]         = water / (cement + 1e-6)

    # ── Binder composition ratios ─────────────────────────────────────────────
    d["scm_ratio"]        = (fa + sf + ggbfs) / (total_binder + 1e-6)
    d["cement_ratio"]     = cement / (total_binder + 1e-6)
    d["ggbfs_ratio"]      = ggbfs / (total_binder + 1e-6)
    d["fa_ratio"]         = fa / (total_binder + 1e-6)
    d["fine_agg_ratio"]   = fine / (total_agg + 1e-6)

    # ── SP efficiency ratios ──────────────────────────────────────────────────
    d["sp_binder"]        = sp / (total_binder + 1e-6)
    d["sp_water"]         = sp / (water + 1e-6)

    # ── Volume fractions ──────────────────────────────────────────────────────
    d["binder_density"]   = total_binder / (total_vol + 1e-6)
    d["water_vol_ratio"]  = water / (total_vol + 1e-6)
    d["paste_vol"]        = (total_binder + water) / (total_vol + 1e-6)

    # ── Log transforms (linearise skewed distributions) ───────────────────────
    d["log_cement"]       = np.log1p(cement)
    d["log_binder"]       = np.log1p(total_binder)
    d["log_water"]        = np.log1p(water)
    d["log_wc"]           = np.log1p(water / (cement + 1e-6))
    d["log_wb"]           = np.log1p(d["WB_Ratio"])

    # ── Polynomial terms for dominant predictors ──────────────────────────────
    d["cement_sq"]        = cement ** 2
    d["water_sq"]         = water ** 2
    d["sp_sq"]            = sp ** 2
    d["wb_sq"]            = d["WB_Ratio"] ** 2
    d["wc_sq"]            = d["wc_ratio"] ** 2
    d["binder_sq"]        = total_binder ** 2

    # ── Cross/interaction terms ───────────────────────────────────────────────
    d["cement_x_sp"]      = cement * sp
    d["cement_x_sf"]      = cement * sf
    d["cement_x_ggbfs"]   = cement * ggbfs
    d["cement_x_water"]   = cement * water
    d["cement_x_fa"]      = cement * fa
    d["sp_x_water"]       = sp * water
    d["binder_x_wb"]      = total_binder * d["WB_Ratio"]
    d["sf_x_wb"]          = sf * d["WB_Ratio"]
    d["ggbfs_x_wb"]       = ggbfs * d["WB_Ratio"]

    # ── Ratio² terms ──────────────────────────────────────────────────────────
    d["scm_sq"]           = d["scm_ratio"] ** 2
    d["ggbfs_ratio_sq"]   = d["ggbfs_ratio"] ** 2
    d["fa_ratio_sq"]      = d["fa_ratio"] ** 2

    return d


def get_feature_cols(base_cols):
    """Return all feature column names produced by engineer_features."""
    dummy = pd.DataFrame(np.zeros((1, len(base_cols))), columns=base_cols)
    dummy = engineer_features(dummy)
    return list(dummy.columns)


# ============================================================
# LOAD DATA + TRAIN MODELS
# ============================================================

@st.cache_resource
def load_models():
    # ── Load training data ───────────────────────────────────────────────────
    train_df = pd.read_excel(TRAIN_FILE)
    for col in train_df.columns:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
    train_df.dropna(inplace=True)

    # ── Mild outlier removal (keep wide range, 2.5×IQR on 10-90 pct) ─────────
    Q1  = train_df[TARGET].quantile(0.10)
    Q3  = train_df[TARGET].quantile(0.90)
    IQR = Q3 - Q1
    train_df = train_df[
        (train_df[TARGET] >= Q1 - 2.5 * IQR) &
        (train_df[TARGET] <= Q3 + 2.5 * IQR)
    ].reset_index(drop=True)

    # ── Feature engineering ──────────────────────────────────────────────────
    train_df = engineer_features(train_df)
    all_features = get_feature_cols(BASE_COLS)

    X_train = train_df[all_features].copy()
    y_train = train_df[TARGET].copy()
    y_train_log = np.log1p(y_train)

    # ── Scaling ───────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=all_features)

    # ────────────────────────────────────────────────────────────────────────
    # BASE LEARNERS
    # ────────────────────────────────────────────────────────────────────────

    rf_model = RandomForestRegressor(
        n_estimators=600,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=0.5,
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )

    xgb_model = xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.008,
        max_depth=6,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.75,
        colsample_bylevel=0.75,
        reg_alpha=0.05,
        reg_lambda=1.0,
        gamma=0.05,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )

    lgb_model = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.008,
        max_depth=6,
        num_leaves=50,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=0.05,
        reg_lambda=1.0,
        min_child_samples=5,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    # ────────────────────────────────────────────────────────────────────────
    # STACKING ENSEMBLE  (RF + XGB + LGB  →  Ridge meta-learner)
    # ────────────────────────────────────────────────────────────────────────

    stacking_model = StackingRegressor(
        estimators=[
            ("rf",  rf_model),
            ("xgb", xgb_model),
            ("lgb", lgb_model),
        ],
        final_estimator=Ridge(alpha=0.5),
        cv=5,
        passthrough=False,
        n_jobs=-1,
    )

    stacking_model.fit(X_train_sc, y_train_log)

    # ── Also fit individual models for comparison tab ─────────────────────────
    rf_ind = RandomForestRegressor(
        n_estimators=600, max_features=0.5, min_samples_leaf=1,
        random_state=42, n_jobs=-1,
    )
    xgb_ind = xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.01, max_depth=6,
        subsample=0.85, colsample_bytree=0.75, tree_method="hist",
        n_jobs=-1, random_state=42, verbosity=0,
    )
    lgb_ind = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.01, max_depth=6,
        num_leaves=50, subsample=0.85, colsample_bytree=0.75,
        random_state=42, n_jobs=-1, verbose=-1,
    )

    for m in [rf_ind, xgb_ind, lgb_ind]:
        m.fit(X_train_sc, y_train_log)

    return (
        stacking_model, rf_ind, xgb_ind, lgb_ind,
        train_df, all_features, scaler,
    )


(stacking_model, rf_model, xgb_model, lgb_model,
 base_df, all_features, scaler) = load_models()


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():
    test_df = pd.read_excel(TEST_FILE)
    test_df = test_df.rename(columns=TEST_COL_MAP)

    for col in BASE_COLS + [TARGET]:
        if col in test_df.columns:
            test_df[col] = pd.to_numeric(test_df[col], errors="coerce")
    test_df.dropna(subset=BASE_COLS + [TARGET], inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    test_eng  = engineer_features(test_df)
    X_test    = test_eng[all_features].copy()
    y_test    = test_df[TARGET].copy()
    X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=all_features)

    return X_test_sc, y_test, test_df[BASE_COLS]


X_test_sc, y_test, X_test_original = load_test_data()


# ============================================================
# HELPERS
# ============================================================

def predict_strength(model, X_scaled):
    return np.expm1(model.predict(X_scaled))


# ============================================================
# OPTIMISATION GRID
# ============================================================

total_binder_mean = base_df[["Cement", "FlyAsh", "SilicaFume", "GGBFS"]].sum(axis=1).mean()
total_agg_mean    = base_df[["FineAgg", "CoarseAgg"]].sum(axis=1).mean()
water_mean        = base_df["Water"].mean()
base_sp           = base_df["SP"].mean()

fa_range         = np.linspace(0,    0.30, 5)
sf_range         = np.linspace(0.02, 0.10, 4)
ggbfs_range      = np.linspace(0.05, 0.60, 6)
fine_ratio_range = np.linspace(0.35, 0.45, 4)
sp_range         = np.linspace(base_sp * 0.8, base_sp * 1.3, 5)
wb_range         = np.linspace(0.28, 0.55, 5)


def optimize(target_strength, cost):
    fa_g, sf_g, ggbfs_g, fine_g, sp_g, wb_g = np.meshgrid(
        fa_range, sf_range, ggbfs_range,
        fine_ratio_range, sp_range, wb_range, indexing="ij"
    )

    fa_f    = fa_g.ravel()
    sf_f    = sf_g.ravel()
    ggbfs_f = ggbfs_g.ravel()
    fine_f  = fine_g.ravel()
    sp_f    = sp_g.ravel()
    wb_f    = wb_g.ravel()

    total_scm = fa_f + sf_f + ggbfs_f
    mask = total_scm <= 0.75
    fa_f, sf_f, ggbfs_f, fine_f, sp_f, wb_f = (
        arr[mask] for arr in (fa_f, sf_f, ggbfs_f, fine_f, sp_f, wb_f)
    )
    total_scm = total_scm[mask]

    cement = total_binder_mean * (1 - total_scm)
    fa     = total_binder_mean * fa_f
    sf     = total_binder_mean * sf_f
    ggbfs  = total_binder_mean * ggbfs_f
    fine   = total_agg_mean * fine_f
    coarse = total_agg_mean * (1 - fine_f)
    sp     = sp_f
    water  = wb_f * (cement + fa + sf + ggbfs)

    raw = pd.DataFrame({
        "Cement": cement, "Water": water,
        "CoarseAgg": coarse, "FineAgg": fine,
        "FlyAsh": fa, "SilicaFume": sf,
        "GGBFS": ggbfs, "SP": sp,
    })
    raw_eng   = engineer_features(raw)
    samples_sc = pd.DataFrame(scaler.transform(raw_eng[all_features]), columns=all_features)

    strength = predict_strength(stacking_model, samples_sc)

    mask2 = strength >= target_strength
    if not mask2.any():
        return pd.DataFrame()

    cement, fa, sf, ggbfs, fine, coarse, sp, water, strength = (
        arr[mask2] for arr in (cement, fa, sf, ggbfs, fine, coarse, sp, water, strength)
    )

    total_cost = (
        cement * cost["cement"] + fa     * cost["fa"]     +
        sf     * cost["sf"]     + ggbfs  * cost["ggbfs"]  +
        fine   * cost["fine"]   + coarse * cost["coarse"] +
        sp     * cost["sp"]     + water  * cost["water"]
    )

    co2 = (
        cement * 0.9  + fa     * 0.02  + sf     * 0.1   +
        ggbfs  * 0.07 + fine   * 0.005 + coarse * 0.005 +
        sp     * 0.2  + water  * 0.001
    )

    df_out = pd.DataFrame({
        "Cement (kg)":    cement, "Water (kg)":   water,
        "Fly Ash (kg)":   fa,     "Silica Fume (kg)": sf,
        "GGBFS (kg)":     ggbfs,  "Fine Agg (kg)":    fine,
        "Coarse Agg (kg)":coarse, "SP (kg)":          sp,
        "Strength (MPa)": strength,
        "Cost (₹/m³)":    total_cost,
        "CO₂ (kg/m³)":    co2,
    })

    df_out["Cost Reduction (%)"] = (1 - df_out["Cost (₹/m³)"] / df_out["Cost (₹/m³)"].max()) * 100
    df_out["CO₂ Reduction (%)"]  = (1 - df_out["CO₂ (kg/m³)"] / df_out["CO₂ (kg/m³)"].max()) * 100

    return df_out.sort_values("Cost (₹/m³)").reset_index(drop=True)


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(layout="wide", page_title="AI Concrete Mix Optimizer")
st.title("🏗️ AI Sustainable Concrete Mix Optimizer")
st.caption("RF + XGBoost + LightGBM Stacking · Water-aware features · Log-target · IS 10262:2019 aligned")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("💰 Material Rates (₹/kg)")
cost = {
    "cement": st.sidebar.number_input("Cement",            0.0, 20.0,  8.0),
    "water":  st.sidebar.number_input("Water",             0.0,  5.0,  0.05),
    "fa":     st.sidebar.number_input("Fly Ash",           0.0, 20.0,  2.0),
    "sf":     st.sidebar.number_input("Silica Fume",       0.0, 50.0, 25.0),
    "ggbfs":  st.sidebar.number_input("GGBFS",             0.0, 20.0,  3.5),
    "fine":   st.sidebar.number_input("Fine Aggregate",    0.0, 10.0,  1.2),
    "coarse": st.sidebar.number_input("Coarse Aggregate",  0.0, 10.0,  1.5),
    "sp":     st.sidebar.number_input("Superplasticizer",  0.0,100.0, 60.0),
}
target_strength = st.sidebar.slider("Target Strength (MPa)", 20, 100, 40)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_opt, tab_perf, tab_shap = st.tabs(
    ["🔍 Optimize Mix", "🧪 Model Performance", "📊 SHAP Explainability"]
)


# ── TAB 1 : OPTIMISATION ──────────────────────────────────────────────────────
with tab_opt:
    if st.button("🔍 Optimize Mix", type="primary"):
        with st.spinner("Running vectorised optimisation…"):
            results = optimize(target_strength, cost)

        if results.empty:
            st.warning("No feasible mix found. Try lowering the target strength.")
        else:
            best = results.iloc[0]

            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Strength", f"{best['Strength (MPa)']:.1f} MPa")
            col2.metric("Cost",               f"₹{best['Cost (₹/m³)']:.0f}/m³")
            col3.metric("CO₂ Footprint",      f"{best['CO₂ (kg/m³)']:.1f} kg/m³")

            st.subheader("🏆 Best Mix Proportions")
            mix_cols = ["Cement (kg)", "Water (kg)", "Fly Ash (kg)", "Silica Fume (kg)",
                        "GGBFS (kg)", "Fine Agg (kg)", "Coarse Agg (kg)", "SP (kg)"]
            st.dataframe(best[mix_cols].to_frame("kg/m³").T, use_container_width=True)

            fig1 = px.scatter(
                results, x="Cost (₹/m³)", y="Strength (MPa)", color="CO₂ (kg/m³)",
                color_continuous_scale="RdYlGn_r",
                title="Pareto Front — Strength vs Cost (coloured by CO₂)",
                hover_data=["Cement (kg)", "GGBFS (kg)", "Silica Fume (kg)", "Water (kg)"],
            )
            st.plotly_chart(fig1, use_container_width=True)

            st.subheader("📋 Top 10 Mixes")
            st.dataframe(results.head(10), use_container_width=True)


# ── TAB 2 : MODEL PERFORMANCE ─────────────────────────────────────────────────
with tab_perf:
    st.subheader("Model Testing Performance on 20% Hold-out Dataset")

    stack_pred = predict_strength(stacking_model, X_test_sc)
    rf_pred    = np.expm1(rf_model.predict(X_test_sc))
    xgb_pred   = np.expm1(xgb_model.predict(X_test_sc))
    lgb_pred   = np.expm1(lgb_model.predict(X_test_sc))

    perf = pd.DataFrame({
        "Model": ["Stacking Ensemble ⭐", "Random Forest", "XGBoost", "LightGBM"],
        "R² Score": [
            r2_score(y_test, stack_pred),
            r2_score(y_test, rf_pred),
            r2_score(y_test, xgb_pred),
            r2_score(y_test, lgb_pred),
        ],
        "MAE (MPa)": [
            mean_absolute_error(y_test, stack_pred),
            mean_absolute_error(y_test, rf_pred),
            mean_absolute_error(y_test, xgb_pred),
            mean_absolute_error(y_test, lgb_pred),
        ],
        "RMSE (MPa)": [
            np.sqrt(mean_squared_error(y_test, stack_pred)),
            np.sqrt(mean_squared_error(y_test, rf_pred)),
            np.sqrt(mean_squared_error(y_test, xgb_pred)),
            np.sqrt(mean_squared_error(y_test, lgb_pred)),
        ],
    })
    perf["R² Score"]   = perf["R² Score"].map("{:.4f}".format)
    perf["MAE (MPa)"]  = perf["MAE (MPa)"].map("{:.2f}".format)
    perf["RMSE (MPa)"] = perf["RMSE (MPa)"].map("{:.2f}".format)
    st.dataframe(perf, use_container_width=True)

    st.subheader("Actual vs Predicted Strength — Stacking Ensemble")
    cmp = pd.DataFrame({
        "Actual Strength (MPa)":    y_test.values,
        "Predicted Strength (MPa)": stack_pred,
    })
    fig2 = px.scatter(
        cmp, x="Actual Strength (MPa)", y="Predicted Strength (MPa)",
        trendline="ols", title="Actual vs Predicted — Stacking Ensemble",
    )
    mn, mx = float(y_test.min()), float(y_test.max())
    fig2.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                   line=dict(dash="dash", color="red"))
    fig2.add_annotation(
        x=0.05, y=0.95, xref="paper", yref="paper",
        text=f"R² = {r2_score(y_test, stack_pred):.4f}",
        showarrow=False, font=dict(size=14),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Residual plot
    residuals = y_test.values - stack_pred
    fig3 = px.scatter(
        x=stack_pred, y=residuals,
        labels={"x": "Predicted Strength (MPa)", "y": "Residual (MPa)"},
        title="Residual Plot — Stacking Ensemble",
    )
    fig3.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig3, use_container_width=True)


# ── TAB 3 : SHAP ──────────────────────────────────────────────────────────────
with tab_shap:
    st.subheader("SHAP Feature Importance — XGBoost (base learner)")
    st.caption("SHAP computed on the XGBoost base learner for TreeExplainer compatibility.")

    try:
        explainer = shap.TreeExplainer(xgb_model)
        shap_vals = explainer.shap_values(X_test_sc.values[:100])

        fig_bar, _ = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_vals, X_test_sc.values[:100],
            feature_names=all_features, plot_type="bar", show=False,
        )
        st.pyplot(fig_bar, bbox_inches="tight")
        plt.close(fig_bar)

        fig_bee, _ = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            shap_vals, X_test_sc.values[:100],
            feature_names=all_features, show=False,
        )
        st.pyplot(fig_bee, bbox_inches="tight")
        plt.close(fig_bee)

    except Exception as e:
        st.warning(f"SHAP plot could not be generated: {e}")
