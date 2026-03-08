import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ============================================================
# LOAD DATA AND TRAIN MODEL
# ============================================================

@st.cache_resource
def load_model():

    df = pd.read_excel("dataset_80_percent.xlsx").iloc[1:].reset_index(drop=True)

    ID_COL = "Column1"
    CEMENT = "Column2"
    FINE = "Column3"
    COARSE = "Column4"
    FA = "Column6"
    SF = "Column7"
    GGBFS = "Column8"
    SP = "Column9"
    TARGET = "Column10"

    for col in df.columns:
        if col != ID_COL:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)

    feature_cols = [CEMENT, FINE, COARSE, FA, SF, GGBFS, SP]

    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)

    return model, df, feature_cols, r2


model, base_df, feature_cols, r2 = load_model()

# ============================================================
# BASE VALUES
# ============================================================

total_binder = base_df[['Column2','Column6','Column7','Column8']].sum(axis=1).mean()
total_agg = base_df[['Column3','Column4']].sum(axis=1).mean()
base_sp = base_df['Column9'].mean()

# Extended ranges (important for high strength)

fa_range = np.linspace(0,0.30,7)
sf_range = np.linspace(0,0.12,6)
ggbfs_range = np.linspace(0.15,0.65,9)

fine_ratio_range = np.linspace(0.35,0.50,6)

sp_range = np.linspace(base_sp*0.7, base_sp*1.7,6)

# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(layout="wide")
st.title("🏗️ AI Sustainable Concrete Optimizer")

st.success(f"Model Accuracy (R²): {r2:.3f}")

# ============================================================
# USER INPUTS
# ============================================================

st.sidebar.header("Material Cost ₹/kg")

cost_inputs = {
    'cement': st.sidebar.number_input("Cement", value=8.0),
    'fly_ash': st.sidebar.number_input("Fly Ash", value=2.0),
    'silica_fume': st.sidebar.number_input("Silica Fume", value=25.0),
    'ggbfs': st.sidebar.number_input("GGBFS", value=3.5),
    'fine_aggregate': st.sidebar.number_input("Fine Aggregate", value=1.2),
    'coarse_aggregate': st.sidebar.number_input("Coarse Aggregate", value=1.5),
    'superplasticizer': st.sidebar.number_input("Superplasticizer", value=60.0)
}

target_strength = st.sidebar.slider("Target Strength MPa", 30, 70, 40)

dataset_max = base_df["Column10"].max()

if target_strength > dataset_max + 5:
    st.warning("Target strength may be outside training range.")

# ============================================================
# SAFE PREDICTION FUNCTION
# ============================================================

def safe_predict(sample):

    try:
        pred = model.predict(sample)[0]

        if np.isnan(pred) or pred < 0:
            return None

        return pred

    except:
        return None

# ============================================================
# BASELINE MIX
# ============================================================

def calculate_baseline():

    cement = total_binder
    fine = total_agg * 0.4
    coarse = total_agg * 0.6
    sp = base_sp

    sample = pd.DataFrame(
        [[cement, fine, coarse, 0, 0, 0, sp]],
        columns=feature_cols
    )

    strength = safe_predict(sample)

    cost = (
        cement * cost_inputs['cement'] +
        fine * cost_inputs['fine_aggregate'] +
        coarse * cost_inputs['coarse_aggregate'] +
        sp * cost_inputs['superplasticizer']
    )

    co2 = cement * 0.9 + sp * 0.2

    return strength, cost, co2


baseline_strength, baseline_cost, baseline_co2 = calculate_baseline()

# ============================================================
# OPTIMIZATION
# ============================================================

def optimize():

    results = []

    for ggbfs_p in ggbfs_range:
        for fa_p in fa_range:
            for sf_p in sf_range:

                scm = fa_p + sf_p + ggbfs_p

                if scm > 0.75:
                    continue

                cement = total_binder * (1 - scm)
                fa = total_binder * fa_p
                sf = total_binder * sf_p
                ggbfs = total_binder * ggbfs_p

                for fine_ratio in fine_ratio_range:

                    coarse_ratio = 1 - fine_ratio

                    fine = total_agg * fine_ratio
                    coarse = total_agg * coarse_ratio

                    for sp in sp_range:

                        sample = pd.DataFrame(
                            [[cement,fine,coarse,fa,sf,ggbfs,sp]],
                            columns=feature_cols
                        )

                        strength = safe_predict(sample)

                        if strength is None:
                            continue

                        if strength < target_strength:
                            continue

                        cost = (
                            cement*cost_inputs['cement']
                            + fa*cost_inputs['fly_ash']
                            + sf*cost_inputs['silica_fume']
                            + ggbfs*cost_inputs['ggbfs']
                            + fine*cost_inputs['fine_aggregate']
                            + coarse*cost_inputs['coarse_aggregate']
                            + sp*cost_inputs['superplasticizer']
                        )

                        co2 = (
                            cement*0.9
                            + fa*0.02
                            + sf*0.1
                            + ggbfs*0.07
                            + fine*0.005
                            + coarse*0.005
                            + sp*0.2
                        )

                        results.append({

                            "Cement":cement,
                            "FlyAsh":fa,
                            "SilicaFume":sf,
                            "GGBFS":ggbfs,
                            "FineAgg":fine,
                            "CoarseAgg":coarse,
                            "SP":sp,
                            "Strength":strength,
                            "Cost":cost,
                            "CO2":co2,
                            "CostReduction":(1-cost/baseline_cost)*100,
                            "CO2Reduction":(1-co2/baseline_co2)*100
                        })

    df = pd.DataFrame(results)

    if df.empty:
        return df

    df = df.sort_values("CostReduction", ascending=False)

    return df.head(10)

# ============================================================
# RUN BUTTON
# ============================================================

if st.button("🔍 Optimize Mix"):

    with st.spinner("Searching mixes..."):

        results = optimize()

    if results.empty:

        st.error(
            "No feasible mix found.\n\n"
            "Possible reasons:\n"
            "• Target strength too high\n"
            "• SCM ratios too high\n"
            "• Model prediction uncertainty"
        )

    else:

        st.subheader("🏆 Best Mix")

        best = results.iloc[0]

        st.write(best)

        st.subheader("Top 10 Mix Designs")

        st.dataframe(results.round(2))

        fig = px.scatter(
            results,
            x="Cost",
            y="Strength",
            size="CO2Reduction",
            color="CostReduction"
        )

        st.plotly_chart(fig)

st.caption("Run using: streamlit run app.py")