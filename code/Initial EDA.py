# ================================================
# EDA: Food Access Dataset
# ================================================
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# Load dataset
file_path = "D:\Data Science\Data Mining\cleaned dataset\Cleaned_Food_Access_Dataset.csv"
df = pd.read_csv(file_path)

print("\n✅ Dataset Loaded.")
print("Shape of the dataset:", df.shape)

#%%
# Data types and missing values
print("\n🔎 Data Types and Missing Value Summary:")
print(df.info())

missing_counts = df.isnull().sum()
missing_percent = (missing_counts / len(df)) * 100
missing_summary = pd.DataFrame({
    "Missing Count": missing_counts,
    "Missing %": missing_percent
}).sort_values(by="Missing %", ascending=False)

print("\n🧹 Columns with Missing Values:")
print(missing_summary[missing_summary["Missing Count"] > 0])

#%%
# Summary statistics
print("\n📊 Summary Statistics:")
print(df.describe())

# %%
# Histogram: lapop10
plt.figure(figsize=(8, 5))
sns.histplot(df['lapop10'].dropna(), bins=30, kde=True, color='skyblue')
plt.title("Distribution of % Population with Low Food Access (lapop10)")
plt.xlabel("% Population with Low Access (10 miles)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("lapop10_hist.png")
plt.show()

# %%
# Correlation Heatmap: Key Variables
selected_cols = ['lapop10', 'PovertyRate', 'MedianFamilyIncome', 'TractSNAP', 'TractHUNV']
plt.figure(figsize=(6, 5))
sns.heatmap(df[selected_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Heatmap – Key Food Access Variables")
plt.tight_layout()
plt.savefig("food_access_corr_heatmap.png")
plt.show()
# %%
