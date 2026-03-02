import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import ExtraTreesRegressor

# ============================================================
# LOAD AND TRAIN MODEL
# ============================================================

@st.cache_resource
def load_model():
    train_df = pd.read_excel("dataset_80_percent.xlsx").iloc[1:].reset_index(drop=True)

    ID_COL = "Column1"
    CEMENT_COL = "Column2"
    FINE_AGG_COL = "Column3"
    COARSE_AGG_COL = "Column4"
    FA_COL = "Column6"
    SF_COL = "Column7"
    GGBFS_COL = "Column8"
    SP_COL = "Column9"
    TARGET = "Column10"

    for col in train_df.columns:
        if col != ID_COL:
            train_df[col] = pd.to_numeric(train_df[col], errors="coerce")

    train_df.dropna(inplace=True)

    feature_cols = [
        CEMENT_COL,
        FINE_AGG_COL,
        COARSE_AGG_COL,
        FA_COL,
        SF_COL,
        GGBFS_COL,
        SP_COL
    ]

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET]

    model = ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    return model, train_df, feature_cols


model, base_df, feature_cols = load_model()

# ============================================================
# BASE VALUES
# ============================================================

total_binder = base_df[['Column2','Column6','Column7','Column8']].sum(axis=1).mean()
total_agg = base_df[['Column3','Column4']].sum(axis=1).mean()
base_sp = base_df['Column9'].mean()

fa_range = np.linspace(0, 0.30, 6)
sf_range = np.linspace(0, 0.10, 5)
ggbfs_range = np.linspace(0.20, 0.60, 8)
fine_ratio_range = np.linspace(0.35, 0.45, 4)
sp_range = np.linspace(base_sp*0.8, base_sp*1.5, 4)

# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="AI Sustainable Concrete Optimizer", layout="wide")
st.title("🏗️ AI Sustainable Concrete Mix Optimizer")

st.sidebar.header("💰 Enter Local Rates (₹/kg)")

cost_inputs = {
    'cement': st.sidebar.number_input("Cement", value=8.0),
    'fly_ash': st.sidebar.number_input("Fly Ash", value=2.0),
    'silica_fume': st.sidebar.number_input("Silica Fume", value=25.0),
    'ggbfs': st.sidebar.number_input("GGBFS", value=3.5),
    'fine_aggregate': st.sidebar.number_input("Fine Aggregate", value=1.2),
    'coarse_aggregate': st.sidebar.number_input("Coarse Aggregate", value=1.5),
    'superplasticizer': st.sidebar.number_input("Superplasticizer", value=60.0)
}

target_strength = st.sidebar.slider("Target Strength (MPa)", 30, 60, 40)

# ============================================================
# BASELINE MIX (100% CEMENT)
# ============================================================

def calculate_baseline():
    cement = total_binder
    fine = total_agg * 0.4
    coarse = total_agg * 0.6
    sp = base_sp

    sample = pd.DataFrame([[cement, fine, coarse, 0, 0, 0, sp]], columns=feature_cols)
    strength = model.predict(sample)[0]

    cost = (
        cement * cost_inputs['cement'] +
        fine * cost_inputs['fine_aggregate'] +
        coarse * cost_inputs['coarse_aggregate'] +
        sp * cost_inputs['superplasticizer']
    )

    co2 = cement * 0.90 + sp * 0.20

    return strength, cost, co2


baseline_strength, baseline_cost, baseline_co2 = calculate_baseline()

# ============================================================
# OPTIMIZATION FUNCTION
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

                        strength = model.predict(sample)[0]

                        if strength < target_strength:
                            continue

                        cost = (
                            cement * cost_inputs['cement'] +
                            fa * cost_inputs['fly_ash'] +
                            sf * cost_inputs['silica_fume'] +
                            ggbfs * cost_inputs['ggbfs'] +
                            fine * cost_inputs['fine_aggregate'] +
                            coarse * cost_inputs['coarse_aggregate'] +
                            sp * cost_inputs['superplasticizer']
                        )

                        co2 = (
                            cement * 0.90 +
                            fa * 0.02 +
                            sf * 0.10 +
                            ggbfs * 0.07 +
                            fine * 0.005 +
                            coarse * 0.005 +
                            sp * 0.20
                        )

                        results.append({
                            "Cement (kg/m³)": cement,
                            "Fly Ash (kg/m³)": fa,
                            "Silica Fume (kg/m³)": sf,
                            "GGBFS (kg/m³)": ggbfs,
                            "Fine Aggregate (kg/m³)": fine,
                            "Coarse Aggregate (kg/m³)": coarse,
                            "Superplasticizer (kg/m³)": sp,
                            "Strength (MPa)": strength,
                            "Cost (₹/m³)": cost,
                            "CO2 (kg/m³)": co2,
                            "Cost Reduction (%)": (1 - cost/baseline_cost)*100,
                            "CO2 Reduction (%)": (1 - co2/baseline_co2)*100
                        })

    df = pd.DataFrame(results)

    if df.empty:
        return df

    df = df.sort_values(by="Cost Reduction (%)", ascending=False)
    return df.head(10)


# ============================================================
# RUN BUTTON
# ============================================================

if st.button("🔍 Optimize Mix"):

    with st.spinner("Optimizing sustainable mixes... Please wait ⏳"):
        results = optimize()

    if results.empty:
        st.warning("No feasible mix found. Try lowering target strength.")
    else:
        st.subheader("🏆 OPTIMUM MIX DESIGN (Per m³)")

        best = results.iloc[0]

        st.write("### Material Quantities:")
        st.write(f"Cement: {best['Cement (kg/m³)']:.1f} kg")
        st.write(f"Fly Ash: {best['Fly Ash (kg/m³)']:.1f} kg")
        st.write(f"Silica Fume: {best['Silica Fume (kg/m³)']:.1f} kg")
        st.write(f"GGBFS: {best['GGBFS (kg/m³)']:.1f} kg")
        st.write(f"Fine Aggregate: {best['Fine Aggregate (kg/m³)']:.1f} kg")
        st.write(f"Coarse Aggregate: {best['Coarse Aggregate (kg/m³)']:.1f} kg")
        st.write(f"Superplasticizer: {best['Superplasticizer (kg/m³)']:.2f} kg")

        st.write("### Performance:")
        st.write(f"Strength: {best['Strength (MPa)']:.2f} MPa")
        st.write(f"Cost: ₹{best['Cost (₹/m³)']:.2f}")
        st.write(f"CO2: {best['CO2 (kg/m³)']:.2f} kg")

        st.write("### Savings:")
        st.write(f"Cost Reduction: {best['Cost Reduction (%)']:.1f}%")
        st.write(f"CO2 Reduction: {best['CO2 Reduction (%)']:.1f}%")

        st.subheader("📊 Top 10 Optimized Mixes")
        st.dataframe(results.round(2))

st.caption("Run using: streamlit run app.py")