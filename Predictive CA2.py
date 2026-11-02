
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\Users\himan\OneDrive\Desktop\INT234 CA2\CEA_Statewise_Electricity_Generation_2019-2026.csv")
    
# Display first 5 records
print("\nFIRST 5 RECORDS")
print(df.head())

# Dataset information
print("\nDATASET INFORMATION")
print(df.info())

# Dataset shape
print("\nDATASET SHAPE")
print(df.shape)

# Missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Duplicate records
print("\nDUPLICATE RECORDS")
print(df.duplicated().sum())

# Statistical summary
print("\nSTATISTICAL SUMMARY")
print(df.describe())

# =================================
# EDA PART 2: STATE-WISE ANALYSIS
# =================================

# Total generation by state
state_generation = (df.groupby("state")["actual_generation_mu"].sum().sort_values(ascending=False))
print("\nSTATE-WISE ELECTRICITY GENERATION")
print(state_generation)

# Top 15 states
top_states = state_generation.head(15)

# Visualization
plt.figure(figsize=(12, 7))
top_states.plot(kind="bar")
plt.title("Top 15 States by Total Electricity Generation")
plt.xlabel("State")
plt.ylabel("Total Actual Generation (MU)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# =========================================
# EDA PART 3: YEAR-WISE GENERATION ANALYSIS
# =========================================

# Select complete years only
yearly_data = df[(df["year"] >= 2020) & (df["year"] <= 2025)]
# Calculate yearly generation
year_generation = (yearly_data.groupby("year")["actual_generation_mu"].sum())
print("\nYEAR-WISE ELECTRICITY GENERATION")
print(year_generation)

# Create bar chart
plt.figure(figsize=(10, 6))
year_generation.plot(kind="bar")
plt.title("Year-wise Electricity Generation (2020-2025)")
plt.xlabel("Year")
plt.ylabel("Total Actual Generation (MU)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Create line chart
plt.figure(figsize=(10, 6))
year_generation.plot(kind="line", marker="o")
plt.title("Electricity Generation Trend (2020-2025)")
plt.xlabel("Year")
plt.ylabel("Total Actual Generation (MU)")
plt.grid(True)
plt.tight_layout()
plt.show()



# =========================================
# EDA PART 4: MONTHLY GENERATION ANALYSIS
# =========================================

# Select complete years
monthly_data = df[(df["year"] >= 2020) & (df["year"] <= 2025)]
# Calculate average generation by month
monthly_generation = (monthly_data.groupby("month")["actual_generation_mu"].mean())
  
print("\nMONTH-WISE AVERAGE ELECTRICITY GENERATION")
print(monthly_generation)

# Month names
month_names = ["Jan", "Feb", "Mar", "Apr","May", "Jun", "Jul", "Aug","Sep", "Oct", "Nov", "Dec"]
monthly_generation.index = month_names

# Plot monthly generation
plt.figure(figsize=(12, 6))

monthly_generation.plot(kind="line",marker="o", linewidth=2)
plt.title("Average Monthly Electricity Generation (2020-2025)")
plt.xlabel("Month")
plt.ylabel("Average Actual Generation (MU)")
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================================
# EDA PART 5: REGION-WISE ANALYSIS
# =========================================

# Select complete years
region_data = df[(df["year"] >= 2020) & (df["year"] <= 2025)]
# Calculate region-wise generation
region_generation = (region_data.groupby("region")["actual_generation_mu"].sum().sort_values(ascending=False))
print("\nREGION-WISE ELECTRICITY GENERATION")
print(region_generation)

# Plot region-wise generation
plt.figure(figsize=(10, 6))
region_generation.plot(kind="bar")
plt.title("Region-wise Electricity Generation (2020-2025)")
plt.xlabel("Region")
plt.ylabel("Total Actual Generation (MU)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# =========================================
# EDA PART 6: CORRELATION HEATMAP
# =========================================

# Convert installed capacity into numeric
df["installed_capacity_mw"] = pd.to_numeric(df["installed_capacity_mw"],errors="coerce")
# Select numerical columns
numeric_data = df[["installed_capacity_mw","planned_generation_mu","actual_generation_mu","previous_year_same_month_generation_mu"]]
# Calculate correlation
correlation = numeric_data.corr()
print("\nCORRELATION MATRIX")
print(correlation)

# Create heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(correlation,annot=True,cmap="coolwarm",fmt=".2f",linewidths=0.5)
plt.title("Correlation Heatmap - Electricity Generation")

plt.tight_layout()
plt.show()


# =========================================
# EDA PART 7: FEATURE RELATIONSHIP ANALYSIS
# =========================================

# Plot 1: Planned vs Actual Generation

plt.figure(figsize=(8, 5))

plt.scatter(df["planned_generation_mu"],df["actual_generation_mu"], alpha=0.5)
plt.title("Planned vs Actual Generation")
plt.xlabel("Planned Generation (MU)")
plt.ylabel("Actual Generation (MU)")
plt.grid(True)
plt.tight_layout()
plt.show()


# Plot 2: Installed Capacity vs Actual Generation

plt.figure(figsize=(8, 5))

plt.scatter(df["installed_capacity_mw"],df["actual_generation_mu"],alpha=0.5)
plt.title("Installed Capacity vs Actual Generation")
plt.xlabel("Installed Capacity (MW)")
plt.ylabel("Actual Generation (MU)")
plt.grid(True)
plt.tight_layout()
plt.show()


# Plot 3: Previous Year vs Actual Generation

plt.figure(figsize=(8, 5))

plt.scatter(df["previous_year_same_month_generation_mu"],df["actual_generation_mu"],alpha=0.5)
plt.title("Previous Year vs Actual Generation")
plt.xlabel("Previous Year Generation (MU)")
plt.ylabel("Actual Generation (MU)")
plt.grid(True)

plt.tight_layout()
plt.show()

# =========================================
# FINAL FEATURE ENGINEERING - CORRECTED
# =========================================

# Convert date into datetime
df["date"] = pd.to_datetime(df["date"])

# Sort data
df = df.sort_values(["state", "date"]).copy()
   
# Create numerical month index
df["month_index"] = (df["date"].dt.year * 12 + df["date"].dt.month)
# Previous date month index
df["previous_month_index"] = (df.groupby("state")["month_index"].shift(1))
  
# Previous month generation
df["previous_month_generation"] = (df.groupby("state")["actual_generation_mu"].shift(1))
  
# Check consecutive months
month_gap = (df["month_index"] - df["previous_month_index"])
df.loc[month_gap != 1,"previous_month_generation"] = np.nan
   

# Previous 3 months average
df["rolling_3_month_avg"] = (df.groupby("state")["actual_generation_mu"].transform(lambda x: x.shift(1).rolling(3).mean()))
    
# Check whether 3 previous months are consecutive
df["three_month_gap"] = (df["month_index"] - df.groupby("state")["month_index"].shift(3))
df.loc[df["three_month_gap"] != 3,"rolling_3_month_avg"] = np.nan
   
# Display results
print("\nFINAL FEATURE ENGINEERING RESULT")
print(df[["date","state","actual_generation_mu","previous_month_generation","rolling_3_month_avg"]].head(15))
    
# Check missing values
print("\nMISSING VALUES")
print(df[["previous_month_generation","rolling_3_month_avg"]].isnull().sum())
    
# =========================================
# EDA PART 9: FINAL ML DATASET VALIDATION
# =========================================

# FINAL ML DATASET PREPARATION

ml_df = df[
    [
        "year",
        "month",
        "state",
        "region",
        "installed_capacity_mw",
        "previous_year_same_month_generation_mu",
        "previous_month_generation",
        "rolling_3_month_avg",
        "actual_generation_mu"
    ]
].copy()

# Remove rows with missing values
ml_df = ml_df.dropna()

print("\nFINAL ML DATASET")
print("Dataset Shape:", ml_df.shape)

print("\nFIRST 10 RECORDS")
print(ml_df.head(10))

print("\nMISSING VALUES")
print(ml_df.isnull().sum())

print("\nUNIQUE STATES")
print(ml_df["state"].nunique())

# Save dataset
ml_df.to_csv("Electricity_Generation_ML_Dataset.csv",index=False)
print("\nML DATASET SAVED SUCCESSFULLY!")

#------------------------------------------------

import os

# Save dataset in the same folder as Python file
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"Electricity_Generation_ML_Dataset.csv")
ml_df.to_csv(output_path, index=False)

print("\nML DATASET SAVED SUCCESSFULLY!")
print("Dataset saved at:", output_path)





# =====================================
# STEP 1: SELECT UTTAR PRADESH DATA
# =====================================

up_df = ml_df[ml_df["state"] == "Uttar Pradesh"].copy()

up_df = up_df.sort_values(["year", "month"])

print("\nUTTAR PRADESH DATASET")
print("Dataset Shape:", up_df.shape)

print("\nFIRST 10 RECORDS")
print(up_df.head(10))

print("\nMISSING VALUES")
print(up_df.isnull().sum())

# Save Uttar Pradesh dataset
up_df.to_csv("Uttar_Pradesh_Electricity_ML_Dataset.csv", index=False)
print("\nUTTAR PRADESH DATASET SAVED!")


# =====================================
# STEP 2: LINEAR REGRESSION MODEL
# =====================================


# IMPROVED MACHINE LEARNING MODEL

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv("Electricity_Generation_ML_Dataset.csv")

# Select features and target
X = data.drop(columns=["actual_generation_mu"])
y = data["actual_generation_mu"]

# Categorical and numerical columns
categorical = ["state", "region"]
numerical = [col for col in X.columns if col not in categorical]

# Preprocessing
preprocessor = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore",sparse_output=False), categorical),("num", SimpleImputer(strategy="median"), numerical)])
    
# Time-based split
data = data.sort_values(["year", "month", "state"])
split = int(len(data) * 0.8)

X = data.drop(columns=["actual_generation_mu"])
y = data["actual_generation_mu"]

X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

# Models
models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42
    )
}

results = []

for name, model in models.items():

    pipeline = Pipeline([("preprocessor", preprocessor),("model", model)])
       

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({"Model": name, "MAE":mae, "RMSE":rmse, "R2 Score": r2})
       

comparison = pd.DataFrame(results)

print("\nIMPROVED MODEL COMPARISON")
print(comparison.sort_values("R2 Score", ascending=False))

# ==========================================
# FEATURE IMPORTANCE ANALYSIS - EXTRA TREES
# ==========================================

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

# Create Extra Trees model
best_model = Pipeline([("preprocessor", preprocessor),("model", ExtraTreesRegressor( n_estimators=300,max_depth=15,min_samples_leaf=2,random_state=42, n_jobs=-1))])
    
# Train model
best_model.fit(X_train, y_train)

# Extract feature importance
importance = best_model.named_steps["model"].feature_importances_

# Get feature names
feature_names = best_model.named_steps[ "preprocessor"].get_feature_names_out()
    
# Create DataFrame
feature_importance = pd.DataFrame({"Feature": feature_names,"Importance": importance})

# Sort features
feature_importance = feature_importance.sort_values(by="Importance",ascending=False)
print("\nTOP 15 IMPORTANT FEATURES")
print(feature_importance.head(15))

# Plot graph
plt.figure(figsize=(12, 7))

top_features = feature_importance.head(15).sort_values(by="Importance")
plt.barh(top_features["Feature"],top_features["Importance"])
plt.title("Top 15 Feature Importance - Extra Trees")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
# ==========================================
# ACTUAL VS PREDICTED - ALL 5 MODELS
# ==========================================

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Create figure
plt.figure(figsize=(14, 7))

# Actual values
plt.plot(range(len(y_test)),y_test.values,label="Actual Generation",marker="o",linewidth=2)

# Predictions from all models
for name, model in models.items():

    pipeline = Pipeline([ ("preprocessor", preprocessor),("model", model) ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    plt.plot(range(len(predictions)),predictions,label=name,marker=".",linestyle="--")
        

# Graph formatting
plt.title("Actual vs Predicted Electricity Generation - All Models")
plt.xlabel("Test Data Records")
plt.ylabel("Electricity Generation (MU)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==========================================
# RESIDUAL ANALYSIS - EXTRA TREES MODEL
# ==========================================

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np

# Train Extra Trees model
residual_model = Pipeline([("preprocessor", preprocessor),("model", ExtraTreesRegressor(n_estimators=300,max_depth=15,min_samples_leaf=2,random_state=42,n_jobs=-1))])
residual_model.fit(X_train, y_train)

# Predict test data
residual_pred = residual_model.predict(X_test)

# Calculate residuals
residuals = y_test.values - residual_pred

# Print evaluation
print("\nRESIDUAL ANALYSIS")
print("Mean Residual:", np.mean(residuals))
print("Mean Absolute Residual:", np.mean(np.abs(residuals)))
print("Residual Standard Deviation:", np.std(residuals))

# Create two graphs
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Residual scatter plot
axes[0].scatter(residual_pred, residuals, alpha=0.6)
axes[0].axhline(y=0, linestyle="--")
axes[0].set_title("Residuals vs Predicted Values")
axes[0].set_xlabel("Predicted Generation (MU)")
axes[0].set_ylabel("Residuals (MU)")
axes[0].grid(True)

# Residual distribution
axes[1].hist(residuals, bins=30, edgecolor="black")
axes[1].set_title("Distribution of Residuals")
axes[1].set_xlabel("Residuals (MU)")
axes[1].set_ylabel("Frequency")
axes[1].grid(True)

plt.tight_layout()
plt.show()

