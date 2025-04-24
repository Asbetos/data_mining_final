#%%[markdown]
## Analysis Notebook
# In this notebook, we conduct various analysis to answer our proposed smart questions

#%%
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# %%
df = pd.read_csv('../dataset/FoodAccessResearchAtlasData2019.csv')

# For an initial look at the dataset and its columns:
print("Dataset Columns:")
print(df.columns.tolist())

# Check unique values in the 'Urban' column.
# Typically, 1 indicates urban and 0 indicates rural.
print("\nUnique values in 'Urban' column:", df['Urban'].unique())


# %%[markdown]
# According to the USDA definitions, a "food desert" is typically a low-income tract 
# that also has low access to supermarkets based on established distance criteria.
# Here, we use the 'LILATracts_1And10' column (which applies a 1-mile threshold for urban
# and a 10-mile threshold for rural areas) as an indicator.
# We assume that a value of 1 in 'LILATracts_1And10' indicates that the tract qualifies as a food desert.


#%%[markdown]
## Smart Question 1: Where are food deserts most geographically concentrated across the U.S., and how do these concentrations differ between urban and rural census tracts?


# %%
df['FoodDesert'] = df['LILATracts_1And10']

state_urban_counts = df.groupby(['State', 'Urban'])['FoodDesert'].sum().reset_index()
food_desert_pivot = state_urban_counts.pivot(index='State', columns='Urban', values='FoodDesert').fillna(0)

food_desert_pivot = food_desert_pivot.rename(columns={0: 'Rural Food Desert Count', 1: 'Urban Food Desert Count'})

food_desert_pivot['Total Food Desert Count'] = (
    food_desert_pivot['Urban Food Desert Count'] + food_desert_pivot['Rural Food Desert Count']
)

food_desert_pivot_sorted = food_desert_pivot.sort_values('Total Food Desert Count', ascending=False)
print("\nTop 10 States by Total Food Desert Tract Count:")
display(food_desert_pivot_sorted)

# %%
top_states = food_desert_pivot_sorted.head(10)
plt.figure(figsize=(10, 6))
plt.barh(top_states.index, top_states['Total Food Desert Count'], color='skyblue')
plt.xlabel('Total Food Desert Tract Count')
plt.title('Top 10 States by Food Desert Concentrations')
plt.gca().invert_yaxis()  # Invert y-axis so the highest count appears on top
plt.tight_layout()
plt.show()


# %%
total_tracts_by_state = df.groupby('State').size().reset_index(name='TotalTracts')

# Merge with our aggregated food desert counts.
food_desert_summary = food_desert_pivot_sorted.merge(total_tracts_by_state, left_index=True, right_on='State')
# Calculate the percentage of tracts that are food deserts.
food_desert_summary['FoodDesertPercentage'] = 100 * food_desert_summary['Total Food Desert Count'] / food_desert_summary['TotalTracts']

print("\nFood Desert Summary (Top 10 by count):")
display(food_desert_summary[['State', 'Total Food Desert Count', 'TotalTracts', 'FoodDesertPercentage']])


#%%
state_counts = df.groupby('State')['FoodDesert'].sum().reset_index()
total_counts = df.groupby('State').size().reset_index(name='TotalTracts')
state_counts = state_counts.rename(columns={'FoodDesert': 'TotalFoodDeserts'})

state_abbrev = {
    'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA','Colorado':'CO',
    'Connecticut':'CT','Delaware':'DE','District of Columbia':'DC','Florida':'FL','Georgia':'GA',
    'Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA','Kansas':'KS','Kentucky':'KY',
    'Louisiana':'LA','Maine':'ME','Maryland':'MD','Massachusetts':'MA','Michigan':'MI','Minnesota':'MN',
    'Mississippi':'MS','Missouri':'MO','Montana':'MT','Nebraska':'NE','Nevada':'NV','New Hampshire':'NH',
    'New Jersey':'NJ','New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND',
    'Ohio':'OH','Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI','South Carolina':'SC',
    'South Dakota':'SD','Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA',
    'Washington':'WA','West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY'
}
state_counts['abbr'] = state_counts['State'].map(state_abbrev)
state_counts = state_counts.merge(total_counts, on='State')
state_counts['FoodDesertPercentage'] = 100 * state_counts['TotalFoodDeserts'] / state_counts['TotalTracts']

fig = go.Figure()


fig.add_trace(go.Choropleth(
    locations=state_counts['abbr'],
    z=state_counts['FoodDesertPercentage'],
    locationmode='USA-states',
    colorscale='Reds',
    colorbar_title='% Food Desert Tracts',
    marker_line_color='white'
))


fig.add_trace(go.Scattergeo(
    locations=state_counts['abbr'],
    locationmode='USA-states',
    text=state_counts['FoodDesertPercentage'].apply(lambda x: f"{x:.0f}%"),
    mode='text',
    textfont=dict(size=9, color='black')
))


fig.update_layout(
    title_text='Percentage of Food Desert Tracts by State (2019)',
    geo=dict(
        scope='usa',
        projection=go.layout.geo.Projection(type='albers usa'),
        showlakes=True,
        lakecolor='rgb(255, 255, 255)'
    ),
    width=1200,
    height=800,
    margin={'r':0, 't':50, 'l':0, 'b':0}
)

fig.show()

# %%
group = df.groupby(['State', 'Urban']).agg(
    TotalFoodDeserts=('FoodDesert', 'sum'),
    TotalTracts=('FoodDesert', 'count')
).reset_index()
group['PctFoodDeserts'] = group['TotalFoodDeserts'] / group['TotalTracts'] * 100


urban = group[group['Urban'] == 1].copy()
rural = group[group['Urban'] == 0].copy()


state_abbrev = {
    'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA','Colorado':'CO',
    'Connecticut':'CT','Delaware':'DE','District of Columbia':'DC','Florida':'FL','Georgia':'GA',
    'Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA','Kansas':'KS','Kentucky':'KY',
    'Louisiana':'LA','Maine':'ME','Maryland':'MD','Massachusetts':'MA','Michigan':'MI','Minnesota':'MN',
    'Mississippi':'MS','Missouri':'MO','Montana':'MT','Nebraska':'NE','Nevada':'NV','New Hampshire':'NH',
    'New Jersey':'NJ','New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND',
    'Ohio':'OH','Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI','South Carolina':'SC',
    'South Dakota':'SD','Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA',
    'Washington':'WA','West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY'
}
urban['abbr'] = urban['State'].map(state_abbrev)
rural['abbr'] = rural['State'].map(state_abbrev)
urban['label'] = urban['PctFoodDeserts'].round(0).astype(int).astype(str) + '%'
rural['label'] = rural['PctFoodDeserts'].round(0).astype(int).astype(str) + '%'


fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type':'choropleth'}, {'type':'choropleth'}]],
    subplot_titles=('Urban % Food Desert Tracts', 'Rural % Food Desert Tracts')
)


fig.add_trace(go.Choropleth(
    locations=urban['abbr'],
    z=urban['PctFoodDeserts'],
    locationmode='USA-states',
    colorscale='Blues',
    zmin=0,
    zmax=group['PctFoodDeserts'].max(),
    colorbar=dict(title='% Desert Tracts', len=0.4, y=0.75)
), row=1, col=1)


fig.add_trace(go.Choropleth(
    locations=rural['abbr'],
    z=rural['PctFoodDeserts'],
    locationmode='USA-states',
    colorscale='Greens',
    zmin=0,
    zmax=group['PctFoodDeserts'].max(),
    colorbar=dict(title='% Desert Tracts', len=0.4, y=0.25)
), row=1, col=2)


fig.add_trace(go.Scattergeo(
    locations=urban['abbr'],
    locationmode='USA-states',
    text=urban['label'],
    mode='text',
    textfont=dict(size=10, color='black'),
    geo='geo'
), row=1, col=1)


fig.add_trace(go.Scattergeo(
    locations=rural['abbr'],
    locationmode='USA-states',
    text=rural['label'],
    mode='text',
    textfont=dict(size=10, color='black'),
    geo='geo2'
), row=1, col=2)


fig.update_geos(scope='usa', showlakes=True, lakecolor='white', row=1, col=1)
fig.update_geos(scope='usa', showlakes=True, lakecolor='white', row=1, col=2)


fig.update_layout(
    title_text='Comparison of Food Desert Tract Percentages: Urban vs Rural (2019)',
    width=1200,
    height=700,
    margin={'r':0, 't':80, 'l':0, 'b':0}
)

fig.show()


# %%
urban = group[group['Urban'] == 1].sort_values('PctFoodDeserts', ascending=False)
rural = group[group['Urban'] == 0].sort_values('PctFoodDeserts', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# Urban bar chart
axes[0].barh(urban['State'], urban['PctFoodDeserts'], color='steelblue')
axes[0].invert_yaxis()
axes[0].set_title('Urban % Food Desert Tracts by State (2019)')
axes[0].set_xlabel('% Food Desert Tracts')

# Rural bar chart
axes[1].barh(rural['State'], rural['PctFoodDeserts'], color='seagreen')
axes[1].invert_yaxis()
axes[1].set_title('Rural % Food Desert Tracts by State (2019)')
axes[1].set_xlabel('% Food Desert Tracts')

plt.tight_layout()
plt.show()

# %%[markdown]
# By far Texas has the most food desert tracts, but it also has a large number of total tracts.
# Mississippi has the highest percentage of food desert tracts, with an astonishing 32% of its tracts being food deserts.
# Mississppi also has the highest percentage of food desert tracts in urban areas which contribute to 42% of its total tracts.
# Arizona and alaska share the highest percentage of food desert tracts in rural areas, with 29%.

# %%[markdown]
## Smart Question 2: What demographics and socioeconomic factors are linked to food deserts?


# %%

atlas = pd.read_csv(
    '../dataset/FoodAccessResearchAtlasData2019.csv',
    dtype={'CensusTract': str}
)
atlas['CensusTract'] = atlas['CensusTract'].str.zfill(11)
atlas['CountyFIPS']  = atlas['CensusTract'].str[:5]

# Define binary food‐desert flag
atlas['FoodDesert'] = atlas['LILATracts_1And10'].astype(int)


def load_county_data(path, fips_col='FIPS'):
    df = pd.read_csv(path)
    df['CountyFIPS'] = df[fips_col].astype(str).str.zfill(5)
    return df

soc = load_county_data('../dataset/FE_socioeconomic.csv')
ins = load_county_data('../dataset/FE_insecurity.csv')
hlth = load_county_data('../dataset/FE_health.csv')
stores = load_county_data('../dataset/FE_stores.csv')
restaurants = load_county_data('../dataset/FE_restaurants.csv')


merged_df = (
    atlas

    .merge(soc[['CountyFIPS',
                'POVRATE15','MEDHHINC15','CHILDPOVRATE15',
                'PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10']],
           on='CountyFIPS', how='left')

    .merge(ins[['CountyFIPS','FOODINSEC_12_14']], on='CountyFIPS', how='left')

    .merge(hlth[['CountyFIPS','PCT_OBESE_ADULTS17']], on='CountyFIPS', how='left')

    .merge(stores[['CountyFIPS','GROCPTH16']], on='CountyFIPS', how='left')

    .merge(restaurants[['CountyFIPS','FFRPTH16']], on='CountyFIPS', how='left')
)


# %%
numeric_df = merged_df.select_dtypes(include=[np.number]).dropna()
corr_matrix = numeric_df.corr()

corr_with_fd = corr_matrix['FoodDesert'].drop('FoodDesert')

threshold = 0.4

high_corr = corr_with_fd[ corr_with_fd.abs() > threshold ] \
               .sort_values(key=lambda x: x.abs(), ascending=False)

print("Variables with |corr| >", threshold, "to FoodDesert:\n")
print(high_corr)

# %%
# List of predictor columns
predictors = [
    'LowIncomeTracts','LILATracts_Vehicle','MedianFamilyIncome',
    'lalowihalfshare','lasnap10share','PovertyRate',
    # 'LA1and10'
]

eda_df = merged_df[predictors + ['FoodDesert']].dropna()
corr = eda_df.corr()

plt.figure(figsize=(10,8))
img = plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(img, fraction=0.046, pad=0.04)
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()


# %%
n = len(predictors)
ncols = 3
nrows = math.ceil(n / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*4))
axes = axes.flatten()

# Plot each boxplot in its own subplot
for i, col in enumerate(predictors):
    merged_df.boxplot(column=col, by='FoodDesert', ax=axes[i])
    axes[i].set_title(col)
    axes[i].set_xlabel("FoodDesert")
    axes[i].set_ylabel(col)

# Remove any unused subplots
for j in range(n, len(axes)):
    fig.delaxes(axes[j])

# Clean up the overall figure title and layout
fig.suptitle('')
plt.tight_layout()
plt.show()

# %%[markdown]
# The factors that are most correlated with food deserts are:
# 1. LowIncomeTracts: Number of Census tracts that meet the low-income criterion.
# 2. LILATracts_Vehicle: Tracts classified as low-access based on limited vehicle availability criteria.
# 3. lalowihalfshare: Share of the low-income population living in low-access areas (0.5 mile/10 mile threshold)
# 4. MedianFamilyIncome: Median family income in the Census tract.
# 5. lasnap10share: Share of the population participating in SNAP in low-access areas (10 mile threshold).
# 6. PovertyRate: Percentage of individuals in the tract living below the federal poverty level.
# 7. LA1and10: Total population in low-access areas using a 1 mile (urban) / 10 mile (rural) distance threshold.


#%%[markdown]
## Smart Question 4: Can we develop a predictive model that accurately identifies 
# high-risk areas likely to be or become food deserts based on social and economic indicators?

# %%
X = merged_df[predictors].copy()
X = X.fillna(X.median())  
# X = sm.add_constant(X)     
y = merged_df['FoodDesert']

logit_model = sm.Logit(y, X).fit(disp=False)  

print(logit_model.summary())

# %%

vif_df = pd.DataFrame({
    'feature': X.columns,
    'VIF': [variance_inflation_factor(X.values, i)
            for i in range(X.shape[1])]
})

print("\nVariance Inflation Factors:")
print(vif_df)



#%%
X = merged_df[predictors]
X = X.fillna(X.median())
y = merged_df['FoodDesert']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=75, stratify=y
)
rf = RandomForestClassifier(n_estimators=100, random_state=92)
rf.fit(X_train, y_train)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
auc_rf = roc_auc_score(y_test, y_prob_rf)
print(f"\nRandom Forest ROC AUC: {auc_rf:.3f}")

# Classification report
print("\nRandom Forest Classification Report (threshold=0.5):")
print(classification_report(y_test, (y_prob_rf >= 0.5).astype(int)))

# Feature importance
imp = pd.Series(rf.feature_importances_, index=predictors).sort_values(ascending=False)
plt.figure(figsize=(6,4))
imp.plot(kind='bar', color='teal')
plt.title('Random Forest Feature Importances')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

# %%
