import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import shap

# ============================================================
# LOAD DATA + TRAIN MODELS
# ============================================================

@st.cache_resource
def load_models():

    train_df = pd.read_excel("dataset_80_percent.xlsx").iloc[1:].reset_index(drop=True)

    ID_COL = "Column1"
    TARGET = "Column10"

    feature_cols = [
        "Column2",  # Cement
        "Column3",  # Fine
        "Column4",  # Coarse
        "Column6",  # FA
        "Column7",  # SF
        "Column8",  # GGBFS
        "Column9"   # SP
    ]

    for col in train_df.columns:
        if col != ID_COL:
            train_df[col] = pd.to_numeric(train_df[col], errors="coerce")

    train_df.dropna(inplace=True)

    X = train_df[feature_cols]
    y = train_df[TARGET]

    # Models
    et_model = ExtraTreesRegressor(n_estimators=150, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=150, random_state=42)
    xgb_model = XGBRegressor(n_estimators=150, learning_rate=0.1, max_depth=6)

    et_model.fit(X, y)
    rf_model.fit(X, y)
    xgb_model.fit(X, y)

    return et_model, rf_model, xgb_model, train_df, feature_cols


et_model, rf_model, xgb_model, base_df, feature_cols = load_models()

# ============================================================
# BASE VALUES
# ============================================================

total_binder = base_df[['Column2','Column6','Column7','Column8']].sum(axis=1).mean()
total_agg = base_df[['Column3','Column4']].sum(axis=1).mean()
base_sp = base_df['Column9'].mean()

fa_range = np.linspace(0, 0.30, 4)
sf_range = np.linspace(0, 0.10, 3)
ggbfs_range = np.linspace(0.20, 0.60, 5)
fine_ratio_range = np.linspace(0.35, 0.45, 3)
sp_range = np.linspace(base_sp*0.9, base_sp*1.2, 3)

# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(layout="wide")
st.title("🏗️ AI Sustainable Concrete Mix Optimizer (Advanced Version)")

# Sidebar
st.sidebar.header("💰 Material Rates (₹/kg)")

cost = {
    "cement": st.sidebar.number_input("Cement", 0.0, 20.0, 8.0),
    "fa": st.sidebar.number_input("Fly Ash", 0.0, 20.0, 2.0),
    "sf": st.sidebar.number_input("Silica Fume", 0.0, 50.0, 25.0),
    "ggbfs": st.sidebar.number_input("GGBFS", 0.0, 20.0, 3.5),
    "fine": st.sidebar.number_input("Fine Aggregate", 0.0, 10.0, 1.2),
    "coarse": st.sidebar.number_input("Coarse Aggregate", 0.0, 10.0, 1.5),
    "sp": st.sidebar.number_input("Superplasticizer", 0.0, 100.0, 60.0),
}

target_strength = st.sidebar.slider("Target Strength (MPa)", 30, 60, 40)

# ============================================================
# OPTIMIZATION
# ============================================================

def optimize():

    results = []

    for ggbfs_p in ggbfs_range:
        for fa_p in fa_range:
            for sf_p in sf_range:

                total_scm = fa_p + sf_p + ggbfs_p

                if total_scm > 0.75:
                    continue

                cement = total_binder * (1 - total_scm)
                fa = total_binder * fa_p
                sf = total_binder * sf_p
                ggbfs = total_binder * ggbfs_p

                for fine_ratio in fine_ratio_range:

                    coarse_ratio = 1 - fine_ratio
                    fine = total_agg * fine_ratio
                    coarse = total_agg * coarse_ratio

                    for sp in sp_range:

                        sample = pd.DataFrame(
                            [[cement, fine, coarse, fa, sf, ggbfs, sp]],
                            columns=feature_cols
                        )

                        strength = et_model.predict(sample)[0]

                        if strength < target_strength:
                            continue

                        total_cost = (
                            cement*cost["cement"] +
                            fa*cost["fa"] +
                            sf*cost["sf"] +
                            ggbfs*cost["ggbfs"] +
                            fine*cost["fine"] +
                            coarse*cost["coarse"] +
                            sp*cost["sp"]
                        )

                        co2 = (
                            cement*0.9 +
                            fa*0.02 +
                            sf*0.1 +
                            ggbfs*0.07 +
                            fine*0.005 +
                            coarse*0.005 +
                            sp*0.2
                        )

                        results.append({

                            "Cement": cement,
                            "FA": fa,
                            "SF": sf,
                            "GGBFS": ggbfs,
                            "Fine": fine,
                            "Coarse": coarse,
                            "SP": sp,
                            "Strength": strength,
                            "Cost": total_cost,
                            "CO2": co2

                        })

    df = pd.DataFrame(results)

    if df.empty:
        return df

    df["Cost Reduction (%)"] = (1 - df["Cost"]/df["Cost"].max())*100
    df["CO2 Reduction (%)"] = (1 - df["CO2"]/df["CO2"].max())*100

    return df.sort_values("Cost")

# ============================================================
# RUN
# ============================================================

if st.button("🔍 Optimize Mix"):

    with st.spinner("Optimizing..."):
        results = optimize()

    if results.empty:

        st.warning("No feasible mix found.")

    else:

        st.subheader("🏆 Best Mix")
        best = results.iloc[0]
        st.write(best)

        # ------------------------------------------------
        # Strength vs Cost
        # ------------------------------------------------

        st.subheader("📈 Strength vs Cost")

        fig1 = px.scatter(results, x="Cost", y="Strength", color="CO2")

        st.plotly_chart(fig1, use_container_width=True)

        # ------------------------------------------------
        # Sustainability
        # ------------------------------------------------

        st.subheader("🌱 Sustainability Indicators")

        cement_saved = total_binder - best["Cement"]
        co2_saved = (total_binder*0.9) - best["CO2"]

        col1, col2 = st.columns(2)

        col1.metric("Cement Saved (kg/m³)", f"{cement_saved:.1f}")
        col2.metric("CO2 Saved (kg/m³)", f"{co2_saved:.1f}")

        # ------------------------------------------------
        # Model Comparison
        # ------------------------------------------------

        st.subheader("🤖 Model Comparison")

        X = base_df[feature_cols]
        y = base_df["Column10"]

        et_pred = et_model.predict(X)
        rf_pred = rf_model.predict(X)
        xgb_pred = xgb_model.predict(X)

        comp_df = pd.DataFrame({

            "Model": ["Extra Trees", "Random Forest", "XGBoost"],

            "R2 Score": [
                r2_score(y, et_pred),
                r2_score(y, rf_pred),
                r2_score(y, xgb_pred)
            ],

            "MAE": [
                mean_absolute_error(y, et_pred),
                mean_absolute_error(y, rf_pred),
                mean_absolute_error(y, xgb_pred)
            ]

        })

        st.dataframe(comp_df)

        # ------------------------------------------------
        # SHAP Explainability
        # ------------------------------------------------

        st.subheader("🔍 SHAP Feature Importance")

        explainer = shap.Explainer(et_model)
        shap_values = explainer(base_df[feature_cols].iloc[:100])

        shap.plots.bar(shap_values, show=False)
        st.pyplot(bbox_inches='tight')

        # ------------------------------------------------
        # Sensitivity Analysis
        # ------------------------------------------------

        st.subheader("📊 Sensitivity Analysis (Cement Variation)")

        cement_values = np.linspace(best["Cement"]*0.8, best["Cement"]*1.2, 10)

        sens_strength = []

        for c in cement_values:

            sample = pd.DataFrame(
                [[c, best["Fine"], best["Coarse"],
                  best["FA"], best["SF"], best["GGBFS"], best["SP"]]],
                columns=feature_cols
            )

            sens_strength.append(et_model.predict(sample)[0])

        fig2 = go.Figure()

        fig2.add_trace(
            go.Scatter(x=cement_values, y=sens_strength, mode='lines+markers')
        )

        fig2.update_layout(
            xaxis_title="Cement (kg)",
            yaxis_title="Predicted Strength"
        )

        st.plotly_chart(fig2, use_container_width=True)

        # ------------------------------------------------
        # Top mixes
        # ------------------------------------------------

        st.subheader("📋 Top 10 Mixes")

        st.dataframe(results.head(10))

st.caption("Advanced AI Sustainable Concrete Optimization System")