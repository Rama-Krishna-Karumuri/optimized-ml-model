import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import ExtraTreesRegressor
import shap
import io

# ============================================================
# LOAD AND TRAIN MODEL (Same as your original)
# ============================================================
@st.cache_resource
def load_model():
    train_df = pd.read_excel("dataset_80_percent.xlsx").iloc[1:].reset_index(drop=True)
    test_df = pd.read_excel("dataset_20_percent.xlsx").iloc[1:].reset_index(drop=True)
    
    ID_COL, CEMENT_COL, FINE_AGG_COL, COARSE_AGG_COL, FA_COL, SF_COL, GGBFS_COL, SP_COL, TARGET = "Column1", "Column2", "Column3", "Column4", "Column6", "Column7", "Column8", "Column9", "Column10"
    
    for df in [train_df, test_df]:
        for col in df.columns:
            if col != ID_COL:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(inplace=True)
    
    X_train = train_df.drop(columns=[ID_COL, TARGET])
    y_train = train_df[TARGET]
    
    model = ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, train_df, [CEMENT_COL, FINE_AGG_COL, COARSE_AGG_COL, FA_COL, SF_COL, GGBFS_COL, SP_COL]

model, base_df, feature_cols = load_model()

# Base values
total_binder = base_df[['Column2','Column6','Column7','Column8']].sum(axis=1).mean()
total_agg = base_df[['Column3','Column4']].sum(axis=1).mean()
base_sp = base_df['Column9'].mean()

# Ranges for optimization
fa_range = np.linspace(0, 0.30, 6)
sf_range = np.linspace(0, 0.10, 5)
ggbfs_range = np.linspace(0.20, 0.60, 8)
fine_ratio_range = np.linspace(0.35, 0.45, 4)
sp_range = np.linspace(base_sp*0.8, base_sp*1.5, 4)

# ============================================================
# STREAMLIT APP
# ============================================================
st.set_page_config(page_title="Sustainable Concrete Optimizer", layout="wide")
st.title("🏗️ AI Sustainable Concrete Mix Optimizer")
st.markdown("Enter your **local material rates** and see **reductions** vs traditional mix!")

# Custom Inputs Sidebar
st.sidebar.header("💰 Enter Local Rates (₹/kg)")
cost_inputs = {}
materials = ['Cement', 'Fly Ash (FA)', 'Silica Fume (SF)', 'GGBFS', 'Fine Aggregate', 'Coarse Aggregate', 'Superplasticizer (SP)']
default_costs = [8.0, 2.0, 25.0, 3.5, 1.2, 1.5, 60.0]

for i, mat in enumerate(materials):
    cost_inputs[mat.lower().replace(' ','_')] = st.sidebar.number_input(f"{mat}", value=default_costs[i], min_value=0.0, step=0.1)

st.sidebar.header("⚙️ Design Parameters")
target_strength = st.sidebar.slider("Target Strength (MPa)", 30, 60, 40)
max_co2_factor = st.sidebar.slider("Max CO2 Budget (relative to cement)", 0.5, 1.5, 1.0)

# Calculate Traditional (100% Cement) Baseline
def calculate_baseline():
    traditional_cement = total_binder  # Replace all binders with cement
    traditional_aggs = total_agg * np.array([0.4, 0.6])  # 40-60 fine-coarse
    traditional_sp = base_sp
    
    base_mix = pd.DataFrame([[
        traditional_cement, traditional_aggs[0], traditional_aggs[1], 0, 0, 0, traditional_sp
    ]], columns=feature_cols)
    
    cyl_strength = model.predict(base_mix)[0]  # Cylinder ~0.8*Cube
    cube_strength = cyl_strength / 0.8
    
    base_cost = (traditional_cement * cost_inputs['cement'] + 
                sum(traditional_aggs * [cost_inputs['fine_aggregate'], cost_inputs['coarse_aggregate']]) +
                traditional_sp * cost_inputs['superplasticizer'])
    
    base_co2 = traditional_cement * 0.90 + traditional_sp * 0.20  # Simplified CO2
    
    return {
        'mix': base_mix.iloc[0].to_dict(),
        'cube_strength': cube_strength,
        'cyl_strength': cyl_strength,
        'cost': base_cost,
        'co2': base_co2
    }

baseline = calculate_baseline()

# Optimize Function (Enhanced)
def optimize_with_custom_costs(target_strength, cost_dict, max_co2_factor):
    best_results = []
    
    for ggbfs_p in ggbfs_range:
        for fa_p in fa_range:
            for sf_p in sf_range:
                total_scm = fa_p + sf_p + ggbfs_p
                if total_scm > 0.75: continue
                
                cement = total_binder * (1 - total_scm)
                fa = total_binder * fa_p
                sf = total_binder * sf_p
                ggbfs = total_binder * ggbfs_p
                
                for fine_ratio in fine_ratio_range:
                    coarse_ratio = 1 - fine_ratio
                    fine_agg = total_agg * fine_ratio
                    coarse_agg = total_agg * coarse_ratio
                    
                    for sp_dosage in sp_range:
                        sample = pd.DataFrame([[cement, fine_agg, coarse_agg, fa, sf, ggbfs, sp_dosage]], columns=feature_cols)
                        strength_cyl = model.predict(sample)[0]
                        strength_cube = strength_cyl / 0.8
                        
                        if strength_cube < target_strength: continue
                        
                        total_cost = (cement * cost_dict['cement'] + fa * cost_dict['fly_ash_(fa)'] + 
                                    sf * cost_dict['silica_fume_(sf)'] + ggbfs * cost_dict['ggbfs'] +
                                    fine_agg * cost_dict['fine_aggregate'] + coarse_agg * cost_dict['coarse_aggregate'] +
                                    sp_dosage * cost_dict['superplasticizer_(sp)'])
                        
                        total_co2 = (cement * 0.90 + fa * 0.02 + sf * 0.10 + ggbfs * 0.07 + 
                                   fine_agg * 0.005 + coarse_agg * 0.005 + sp_dosage * 0.20)
                        
                        if total_co2 > baseline['co2'] * max_co2_factor:
                            continue
                        
                        score = total_cost + 0.5 * total_co2
                        best_results.append({
                            'SCM_%': total_scm*100,
                            'Cost': total_cost,
                            'CO2': total_co2,
                            'Cube_Strength': strength_cube,
                            'Cyl_Strength': strength_cyl,
                            'Cost_Reduction_%': (1 - total_cost/baseline['cost'])*100,
                            'CO2_Reduction_%': (1 - total_co2/baseline['co2'])*100,
                            'mix': sample.iloc[0].to_dict()
                        })
    
    return sorted(pd.DataFrame(best_results), key=lambda x: x['Cost_Reduction_%'], reverse=True)[:10]

# Run Optimization
if st.button("🔍 Optimize Sustainable Mixes", type="primary"):
    results_df = optimize_with_custom_costs(target_strength, cost_inputs, max_co2_factor)
    
    if not results_df.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("💰 Max Cost Reduction", f"{results_df['Cost_Reduction_%'].max():.1f}%")
        with col2:
            st.metric("🌿 Max CO2 Reduction", f"{results_df['CO2_Reduction_%'].max():.1f}%")
        with col3:
            st.metric("💪 Avg Cube Strength", f"{results_df['Cube_Strength'].mean():.1f} MPa")
        
        # Graphs
        st.subheader("📈 Reduction vs SCM Content")
        fig = px.scatter(results_df, x='SCM_%', y='Cost_Reduction_%', size='CO2_Reduction_%',
                        color='Cube_Strength', hover_data=['Cyl_Strength'],
                        title="Cost Savings vs Sustainability Trade-offs")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📊 Pareto Front: Cost vs CO2")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=results_df['Cost'], y=results_df['CO2'],
                                mode='markers+text', text=results_df['SCM_%'].round(0),
                                marker=dict(size=results_df['Cube_Strength']/2, color=results_df['Cost_Reduction_%']),
                                name='Optimized Mixes'))
        fig2.add_trace(go.Scatter(x=[baseline['cost']], y=[baseline['co2']], 
                                mode='markers', marker=dict(size=20, symbol='x', color='red'),
                                name='Traditional (100% Cement)'))
        fig2.update_layout(title="Multi-Objective Pareto (Lower=Better)", xaxis_title="Cost ₹/m³", yaxis_title="CO2 kg/m³")
        st.plotly_chart(fig2, use_container_width=True)
        
        # Best Mix Table
        st.subheader("🏆 Top 5 Optimized Mixes")
        st.dataframe(results_df[['SCM_%', 'Cube_Strength', 'Cyl_Strength', 'Cost_Reduction_%', 'CO2_Reduction_%']].round(1).head())
        
        # SHAP Explanation for Best Mix
        st.subheader("🤖 Why This Mix Works (SHAP)")
        best_mix = pd.DataFrame([results_df.iloc[0]['mix']], columns=feature_cols)
        explainer = shap.Explainer(model)
        shap_values = explainer(best_mix)
        st.plotly_chart(shap.plots.waterfall(shap_values[0]), use_container_width=True)
        
        # Best Mix Details
        best = results_df.iloc[0]
        st.markdown(f"""
        ### Optimum Mix (SCM: {best['SCM_%']:.0f}%)
        - **Cube Strength**: {best['Cube_Strength']:.1f} MPa  
        - **Cylinder Strength**: {best['Cyl_Strength']:.1f} MPa
        - **Cost Savings**: {best['Cost_Reduction_%']:.1f}% vs Traditional
        - **CO2 Savings**: {best['CO2_Reduction_%']:.1f}% vs Traditional
        """)
    else:
        st.warning("No feasible mixes found. Try lower target strength or higher CO2 budget!")

# Baseline Display
with st.expander("📋 Traditional Mix Baseline (100% Cement)"):
    st.json({
        'Cube Strength': f"{baseline['cube_strength']:.1f} MPa",
        'Cylinder Strength': f"{baseline['cyl_strength']:.1f} MPa",
        'Cost': f"₹{baseline['cost']:.0f}/m³",
        'CO2': f"{baseline['co2']:.0f} kg/m³"
    })

st.caption("👨‍💻 Run with: `streamlit run app.py` | 📊 Graphs show reductions vs your traditional baseline!")
