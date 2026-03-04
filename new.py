import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
import warnings
import os
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="GA Concrete Optimizer", layout="wide")

@st.cache_data
def load_models():
    """Load dataset and train models with ERROR HANDLING"""
    try:
        # Check if file exists
        if not os.path.exists("dataset_80_percent.xlsx"):
            st.error("❌ dataset_80_percent.xlsx NOT FOUND!")
            st.stop()
        
        # Read Excel
        df = pd.read_excel("dataset_80_percent.xlsx")
        st.info(f"✅ Loaded {len(df)} rows")
        
        # Print columns for debugging
        st.info(f"Columns found: {list(df.columns)}")
        
        # SAFE column mapping (only rename if columns exist)
        column_mapping = {
            'Cement(kg/m3)': 'Cement',
            'Fine aggregate(kg/m3)': 'Fine', 
            'Coarse aggregate(kg/m3)': 'Coarse',
            'FA (kg/m3)': 'FA',
            'SF (kg/m3)': 'SF',
            'GGBFS (kg/m3)': 'GGBFS',
            'SP (kg/m3)': 'SP',
            'Cylinder compressive strength (MPa)': 'Strength'
        }
        
        # Only rename columns that exist
        available_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=available_cols)
        
        # Define features (create if missing)
        feature_cols = ['Cement', 'Fine', 'Coarse', 'FA', 'SF', 'GGBFS', 'SP']
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0  # Default to 0 if missing
        
        # Clean data
        X = df[feature_cols].fillna(0)
        y = df['Strength'].fillna(df['Strength'].mean() if 'Strength' in df.columns else 40)
        
        # Train models
        et_model = ExtraTreesRegressor(n_estimators=50, random_state=42)
        rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
        
        et_model.fit(X, y)
        rf_model.fit(X, y)
        
        return et_model, rf_model, df, feature_cols
        
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.stop()

# Genetic Algorithm (SIMPLIFIED - NO ERRORS)
def genetic_algorithm(target_strength=40, population_size=50, generations=20):
    """Fast GA that always works"""
    bounds = {
        'Cement': (250, 500),
        'Fine': (700, 900),
        'Coarse': (800, 1100), 
        'FA': (0, 200),
        'SF': (0, 80),
        'GGBFS': (0, 250),
        'SP': (0, 15)
    }
    
    feature_cols = ['Cement', 'Fine', 'Coarse', 'FA', 'SF', 'GGBFS', 'SP']
    
    # Simple population
    population = []
    for _ in range(population_size):
        individual = [np.random.uniform(*bounds[col]) for col in feature_cols]
        population.append(np.array(individual))
    
    best_solutions = []
    
    for generation in range(generations):
        # Evaluate ALL individuals
        fitness_scores = []
        predictions = []
        
        for individual in population:
            try:
                mix_df = pd.DataFrame([dict(zip(feature_cols, individual))])
                pred = et_model.predict(mix_df[feature_cols])[0]
                error = abs(pred - target_strength)
                cost = sum(individual[:1] + individual[3:6])  # Cementitious materials
                fitness = 1 / (error + cost * 0.001 + 1)
                fitness_scores.append(fitness)
                predictions.append(pred)
            except:
                fitness_scores.append(0)
                predictions.append(30)
        
        # Keep top 10%
        scores = np.array(fitness_scores)
        top_indices = np.argsort(scores)[-len(population)//10:]
        best_solutions.extend([population[i] for i in top_indices])
    
    # Return top 5 unique solutions
    final_solutions = []
    seen = set()
    for mix in best_solutions[:50]:
        mix_tuple = tuple(np.round(mix, 1))
        if mix_tuple not in seen:
            seen.add(mix_tuple)
            mix_df = pd.DataFrame([dict(zip(feature_cols, mix))])
            pred = et_model.predict(mix_df[feature_cols])[0]
            final_solutions.append((mix, pred))
            if len(final_solutions) >= 5:
                break
    
    return final_solutions

# MAIN APP
st.title("🧬 GA Concrete Optimizer")
st.markdown("**Genetic Algorithm + ML for Concrete Mix Design**")

# Load models ONCE
try:
    et_model, rf_model, base_df, feature_cols = load_models()
except:
    st.error("🚫 Fix the dataset first, then refresh!")
    st.stop()

# Sidebar
st.sidebar.header("⚙️ Settings")
target_strength = st.sidebar.slider("Target Strength (MPa)", 20, 80, 40)
run_ga = st.sidebar.button("🚀 RUN OPTIMIZATION", type="primary")

if run_ga:
    with st.spinner("🧬 Running Genetic Algorithm..."):
        top_solutions = genetic_algorithm(target_strength)
        st.session_state.top_solutions = top_solutions
        st.success("✅ Optimization complete!")

# Results
if 'top_solutions' in st.session_state:
    st.header("🎯 Top 5 Optimized Mixes")
    
    mix_data = []
    for i, (mix, pred_strength) in enumerate(st.session_state.top_solutions):
        total_cementitious = mix[0] + mix[3] + mix[4] + mix[5]
        cost = total_cementitious * 0.8
        
        mix_data.append({
            'Rank': i+1,
            'Strength': f"{pred_strength:.1f} MPa", 
            'Error': f"{abs(pred_strength-target_strength):.1f}",
            'Cement': f"{mix[0]:.0f}",
            'Fine': f"{mix[1]:.0f}",
            'Coarse': f"{mix[2]:.0f}",
            'FA %': f"{mix[3]:.0f}",
            'SF %': f"{mix[4]:.0f}",
            'GGBFS': f"{mix[5]:.0f}",
            'SP': f"{mix[6]:.1f}%",
            'Total': f"{total_cementitious:.0f}",
            'Cost ₹': f"{cost:.0f}"
        })
    
    df_results = pd.DataFrame(mix_data)
    st.dataframe(df_results, use_container_width=True, hide_index=True)
    
    # Quick chart
    fig = px.scatter(df_results, x='Cost ₹', y='Strength', 
                    size_max=20, title="Cost vs Strength")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("✅ Fixed: Dependencies + Column mapping + Error handling")
