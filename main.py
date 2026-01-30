# ==========================================
# OPTIMUM SCM (GGBFS) – FINAL WORKING MODEL
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==========================================
# 1. LOAD DATA (AS-IS)
# ==========================================

train_df = pd.read_excel("dataset_80_percent.xlsx")
test_df  = pd.read_excel("dataset_20_percent.xlsx")

# ==========================================
# 2. DEFINE CORRECT COLUMNS
# ==========================================

ID_COL     = "Column1"   # Mix ID
TARGET     = "Column10"  # Cylinder compressive strength (MPa)
SCM_COL    = "Column8"   # GGBFS (kg/m3)

# ==========================================
# 3. REMOVE NON-NUMERIC HEADER ROW
# ==========================================

# First row contains text → remove it
train_df = train_df.iloc[1:].reset_index(drop=True)
test_df  = test_df.iloc[1:].reset_index(drop=True)

# Convert all except ID to numeric
for col in train_df.columns:
    if col != ID_COL:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        test_df[col]  = pd.to_numeric(test_df[col], errors="coerce")

# Drop rows with missing values
train_df = train_df.dropna()
test_df  = test_df.dropna()

print("\n✅ Data cleaned successfully")
print("Target column:", TARGET)
print("SCM column:", SCM_COL)

# ==========================================
# 4. SPLIT FEATURES & TARGET
# ==========================================

X_train = train_df.drop(columns=[ID_COL, TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[ID_COL, TARGET])
y_test = test_df[TARGET]

# ==========================================
# 5. TRAIN RANDOM FOREST MODEL
# ==========================================

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

print("\n✅ Model training completed")

# ==========================================
# 6. TEST MODEL
# ==========================================

y_pred = model.predict(X_test)

print("\n📊 MODEL PERFORMANCE")
print("R² Score :", r2_score(y_test, y_pred))
print("MAE      :", mean_absolute_error(y_test, y_pred))
print("RMSE     :", np.sqrt(mean_squared_error(y_test, y_pred)))

# ==========================================
# 7. FIND OPTIMUM SCM CONTENT (GGBFS)
# ==========================================

scm_min = train_df[SCM_COL].min()
scm_max = train_df[SCM_COL].max()

scm_range = np.linspace(scm_min, scm_max, 100)

base_sample = X_train.mean().to_frame().T

strengths = []

for scm in scm_range:
    temp = base_sample.copy()
    temp[SCM_COL] = scm
    strengths.append(model.predict(temp)[0])

opt_idx = np.argmax(strengths)

print("\n🏆 OPTIMUM SCM RESULT")
print(f"Optimum GGBFS content = {scm_range[opt_idx]:.2f} kg/m³")
print(f"Maximum strength     = {strengths[opt_idx]:.2f} MPa")

# ==========================================
# 8. PLOT
# ==========================================

plt.figure()
plt.plot(scm_range, strengths)
plt.scatter(scm_range[opt_idx], strengths[opt_idx])
plt.xlabel("GGBFS (kg/m³)")
plt.ylabel("Cylinder Compressive Strength (MPa)")
plt.title("Optimum SCM Content vs Strength")
plt.show()
