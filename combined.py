# ====================================================
# MULTI-SCM OPTIMIZATION (GGBFS-FOCUSED)
# ====================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ====================================================
# 1. LOAD DATA
# ====================================================

train_df = pd.read_excel("dataset_80_percent.xlsx")
test_df  = pd.read_excel("dataset_20_percent.xlsx")

# Column mapping (FIXED)
ID_COL = "Column1"
CEMENT_COL = "Column2"
FA_COL = "Column6"
SF_COL = "Column7"
GGBFS_COL = "Column8"
TARGET = "Column10"

# ====================================================
# 2. CLEAN DATA
# ====================================================

train_df = train_df.iloc[1:].reset_index(drop=True)
test_df  = test_df.iloc[1:].reset_index(drop=True)

for col in train_df.columns:
    if col != ID_COL:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        test_df[col]  = pd.to_numeric(test_df[col], errors="coerce")

train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# ====================================================
# 3. TRAIN MODEL
# ====================================================

X_train = train_df.drop(columns=[ID_COL, TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[ID_COL, TARGET])
y_test = test_df[TARGET]

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

print("✅ Model trained")
print("R² Score:", r2_score(y_test, model.predict(X_test)))

# ====================================================
# 4. GGBFS-FOCUSED MULTI-SCM OPTIMIZATION
# ====================================================

# Total binder (cement + SCM)
total_binder = (
    train_df[CEMENT_COL].mean()
    + train_df[FA_COL].mean()
    + train_df[SF_COL].mean()
    + train_df[GGBFS_COL].mean()
)

# ENGINEERING CONSTRAINTS
fa_range = np.linspace(0, 0.30, 10)       # Fly ash ≤ 30%
sf_range = np.linspace(0, 0.08, 6)        # Silica fume ≤ 8% (STRICT)
ggbfs_range = np.linspace(0.20, 0.70, 20) # GGBFS main SCM (20–70%)

base_mix = X_train.mean().to_frame().T

best_strength = -1
best_mix = None

for ggbfs_p in ggbfs_range:        # 🔴 PRIORITY LOOP
    for fa_p in fa_range:
        for sf_p in sf_range:

            total_scm = fa_p + sf_p + ggbfs_p
            if total_scm > 0.75:   # Practical binder replacement limit
                continue

            cement = total_binder * (1 - total_scm)
            fa = total_binder * fa_p
            sf = total_binder * sf_p
            ggbfs = total_binder * ggbfs_p

            sample = base_mix.copy()
            sample[CEMENT_COL] = cement
            sample[FA_COL] = fa
            sample[SF_COL] = sf
            sample[GGBFS_COL] = ggbfs

            strength = model.predict(sample)[0]

            if strength > best_strength:
                best_strength = strength
                best_mix = (cement, fa, sf, ggbfs)

# ====================================================
# 5. RESULTS
# ====================================================

cement, fa, sf, ggbfs = best_mix

print("\n🏆 OPTIMUM SCM REPLACEMENT (ENGINEERING-CONSTRAINED)")
print(f"Cement        : {cement:.2f} kg/m³")
print(f"Fly Ash       : {fa:.2f} kg/m³")
print(f"Silica Fume   : {sf:.2f} kg/m³  (≤ 8%)")
print(f"GGBFS         : {ggbfs:.2f} kg/m³  (MAIN SCM)")
print(f"Max Strength  : {best_strength:.2f} MPa")

print("\n📌 Replacement Percentages")
print(f"Fly Ash       : {(fa/total_binder)*100:.1f} %")
print(f"Silica Fume   : {(sf/total_binder)*100:.1f} %")
print(f"GGBFS         : {(ggbfs/total_binder)*100:.1f} %")
print(f"Total SCM     : {((fa+sf+ggbfs)/total_binder)*100:.1f} %")
