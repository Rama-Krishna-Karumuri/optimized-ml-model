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
    try:
        train_df = pd.read_excel("dataset_80_percent.xlsx").iloc[1:].reset_index(drop=True)
    except FileNotFoundError:
        st.error("❌ dataset_80_percent.xlsx not found! Please place it in the same folder as this script.")
        st.stop()

    ID_COL = "Column1"
    CEMENT_COL = "Column2"
    FINE_AGG_COL = "Column3"
    COARSE_AGG_COL = "Column4"
    FA_COL = "Column6"
    SF_COL = "Column7"
    GGBFS_COL = "Column8"
    SP_COL = "Column9"
    TARGET = "Column10"

    # Convert columns to numeric
    for col in train_df.columns:
        if col != ID_COL:
            train_df[col] = pd.to_numeric(train_df[col], errors="coerce")

    train_df.dropna(inplace=True)
    
    if train_df.empty:
        st.error("❌ Dataset is empty after cleaning. Please check your data.")
        st.stop()

    feature_cols = [
        CEMENT_COL, FINE_AGG_COL, COARSE_AGG_COL, 
        FA_COL, SF_COL, GGBFS_COL, SP_COL
    ]

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET]

    model = ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    return model, train_df, feature_cols, CEMENT_COL, FINE_AGG_COL, COARSE_AGG_COL, FA_COL, SF_COL, GGBFS_COL, SP_COL

# Load model and data
try:
    model, base_df, feature_cols, CEMENT_COL, FINE_AGG_COL, COARSE_AGG_COL, FA_COL, SF_COL, GGBFS_COL, SP_COL = load_model()
except:
    st.error("Failed to load model. Please check your dataset file.")
    st.stop()

# ============================================================
# BASE VALUES
# ============================================================

total_binder = base_df[[CEMENT_COL, FA_COL, SF_COL, GGBFS_COL]].sum(axis=1).mean()
total_agg = base_df[[FINE_AGG_COL, COARSE_AGG_COL]].sum(axis=1).mean()
base_sp = base_df[SP_COL].mean()

# Define ranges for optimization
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

# Sidebar for inputs
st.sidebar.header("💰 Enter Local Rates (₹/kg)")

cost_inputs = {
    'cement': st.sidebar.number_input("Cement", value=8.0, min_value=0.0),
    'fly_ash': st.sidebar.number_input("Fly Ash", value=2.0, min_value=0.0),
    'silica_fume': st.sidebar.number_input("Silica Fume", value=25.0, min_value=0.0),
    'ggbfs': st.sidebar.number_input("GGBFS", value=3.5, min_value=0.0),
    'fine_aggregate': st.sidebar.number_input("Fine Aggregate", value=1.2, min_value=0.0),
    'coarse_aggregate': st.sidebar.number_input("Coarse Aggregate", value=1.5, min_value=0.0),
    'superplasticizer': st.sidebar.number_input("Superplasticizer", value=60.0, min_value=0.0)
}

target_strength = st.sidebar.slider("🎯 Target Strength (MPa)", 30, 60, 40)

st.sidebar.info(f"""
**Dataset Info:**
- Samples: {len(base_df)}
- Binder: {total_binder:.1f} kg/m³ avg
- Aggregates: {total_agg:.1f} kg/m³ avg
- SP: {base_sp:.2f} kg/m³ avg
""")

# ============================================================
# BASELINE MIX (100% CEMENT)
# ============================================================

@st.cache_data
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

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Baseline Strength", f"{baseline_strength:.1f} MPa")
with col2:
    st.metric("Baseline Cost", f"₹{baseline_cost:.0f}/m³")
with col3:
    st.metric("Baseline CO₂", f"{baseline_co2:.1f} kg/m³")

# ============================================================
# OPTIMIZATION FUNCTION
# ============================================================

@st.cache_data
def optimize(_target_strength, _cost_inputs):
    results = []

    for ggbfs_p in ggbfs_range:
        for fa_p in fa_range:
            for sf_p in sf_range:
                total_scm = fa_p + sf_p + ggbfs_p
                if total_scm > 0.75 or total_scm <= 0:
                    continue

                cement = total_binder * (1 - total_scm)
                fa = total_binder * fa_p
                sf = total_binder * sf_p
                ggbfs = total_binder * ggbfs_p

                if cement < 0:
                    continue

                for fine_ratio in fine_ratio_range:
                    coarse_ratio = 1 - fine_ratio
                    fine = total_agg * fine_ratio
                    coarse = total_agg * coarse_ratio

                    for sp in sp_range:
                        if sp <= 0:
                            continue

                        sample = pd.DataFrame(
                            [[cement, fine, coarse, fa, sf, ggbfs, sp]],
                            columns=feature_cols
                        )

                        strength = model.predict(sample)[0]

                        if strength < _target_strength * 0.95:  # 5% tolerance
                            continue

                        cost = (
                            cement * _cost_inputs['cement'] +
                            fa * _cost_inputs['fly_ash'] +
                            sf * _cost_inputs['silica_fume'] +
                            ggbfs * _cost_inputs['ggbfs'] +
                            fine * _cost_inputs['fine_aggregate'] +
                            coarse * _cost_inputs['coarse_aggregate'] +
                            sp * _cost_inputs['superplasticizer']
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
                            "Cement (kg/m³)": round(cement, 1),
                            "Fly Ash (kg/m³)": round(fa, 1),
                            "Silica Fume (kg/m³)": round(sf, 1),
                            "GGBFS (kg/m³)": round(ggbfs, 1),
                            "Fine Agg (kg/m³)": round(fine, 1),
                            "Coarse Agg (kg/m³)": round(coarse, 1),
                            "SP (kg/m³)": round(sp, 2),
                            "Strength (MPa)": round(strength, 1),
                            "Cost (₹/m³)": round(cost, 0),
                            "CO2 (kg/m³)": round(co2, 1),
                            "Cost Saving (%)": round((1 - cost/baseline_cost)*100, 1),
                            "CO2 Saving (%)": round((1 - co2/baseline_co2)*100, 1)
                        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values(by="Cost Saving (%)", ascending=False).drop_duplicates()
    return df.head(10)

# ============================================================
# RUN OPTIMIZATION
# ============================================================

if st.button("🔍 Optimize Sustainable Mixes", type="primary"):
    with st.spinner("🧠 Optimizing concrete mixes for cost & sustainability..."):
        results_df = optimize(target_strength, cost_inputs)

    if results_df.empty:
        st.error("❌ No feasible mix found!")
        st.info("💡 Try: Lowering target strength, adjusting cost inputs, or checking dataset")
    else:
        st.success(f"✅ Found {len(results_df)} optimized mixes!")

        # Best mix
        st.subheader("🏆 BEST OPTIMIZED MIX")
        best = results_df.iloc[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Strength", f"{best['Strength (MPa)']} MPa", f"{baseline_strength:.1f} MPa")
            st.metric("Cost", f"₹{best['Cost (₹/m³)']}", f"₹{baseline_cost:.0f}")
            st.metric("CO₂", f"{best['CO2 (kg/m³)']} kg", f"{baseline_co2:.1f} kg")
        
        with col2:
            st.metric("💰 Cost Saving", f"{best['Cost Saving (%)']}%")
            st.metric("🌿 CO₂ Saving", f"{best['CO2 Saving (%)']}%")
        
        st.subheader("📋 Material Breakdown (kg/m³)")
        materials = {
            "Cement": best['Cement (kg/m³)'],
            "Fly Ash": best['Fly Ash (kg/m³)'],
            "Silica Fume": best['Silica Fume (kg/m³)'],
            "GGBFS": best['GGBFS (kg/m³)'],
            "Fine Aggregate": best['Fine Agg (kg/m³)'],
            "Coarse Aggregate": best['Coarse Agg (kg/m³)'],
            "Superplasticizer": best['SP (kg/m³)']
        }
        
        st.json(materials)

        # Top 10 table
        st.subheader("📊 Top 10 Optimized Mixes")
        st.dataframe(results_df, use_container_width=True)

# Instructions
with st.expander("📖 How to Use"):
    st.markdown("""
    1. **Place `dataset_80_percent.xlsx`** in the same folder as this script
    2. **Enter local material rates** in ₹/kg (sidebar)
    3. **Set target strength** (30-60 MPa)
    4. **Click Optimize** and get sustainable mix designs!
    
    **Features:**
    - Minimizes cost while meeting strength requirements
    - Maximizes sustainability (CO₂ reduction)
    - Uses GGBFS, Fly Ash, Silica Fume replacements
    - ExtraTreesRegressor ML model trained on your dataset
    """)

st.caption("🚀 Run with: `streamlit run app.py` | Made for civil engineers & sustainability")
