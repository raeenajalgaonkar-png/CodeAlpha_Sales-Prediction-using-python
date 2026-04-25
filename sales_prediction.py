import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

# ══════════════════════════════════════════════════════════════════
# 1. LOAD & EXPLORE DATA
# ══════════════════════════════════════════════════════════════════
df = pd.read_csv('Advertising.csv', index_col=0)
print("=" * 60)
print("SALES PREDICTION – ADVERTISING DATASET")
print("=" * 60)
print(f"\nShape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nBasic stats:\n{df.describe().round(2)}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# ══════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════
df['Total_Spend']      = df['TV'] + df['Radio'] + df['Newspaper']
df['TV_Share']         = df['TV']        / df['Total_Spend']
df['Radio_Share']      = df['Radio']     / df['Total_Spend']
df['TV_Radio_Interaction'] = df['TV'] * df['Radio'] / 1000
df['Log_TV']           = np.log1p(df['TV'])
df['Log_Radio']        = np.log1p(df['Radio'])
df['Log_Newspaper']    = np.log1p(df['Newspaper'])

# Budget segments
df['Budget_Segment'] = pd.cut(df['Total_Spend'],
                               bins=[0, 100, 200, 300, 500],
                               labels=['Low', 'Medium', 'High', 'Very High'])

print(f"\nBudget segment distribution:\n{df['Budget_Segment'].value_counts()}")

# ══════════════════════════════════════════════════════════════════
# 3. FEATURE SELECTION & SPLIT
# ══════════════════════════════════════════════════════════════════
features_base     = ['TV', 'Radio', 'Newspaper']
features_extended = ['TV', 'Radio', 'Newspaper',
                     'TV_Radio_Interaction', 'TV_Share', 'Radio_Share',
                     'Log_TV', 'Log_Radio']

X_base = df[features_base]
X_ext  = df[features_extended]
y      = df['Sales']

X_train_b, X_test_b, y_train, y_test = train_test_split(
    X_base, y, test_size=0.2, random_state=42)
X_train_e, X_test_e, _, _ = train_test_split(
    X_ext, y, test_size=0.2, random_state=42)

scaler  = StandardScaler()
Xtr_b_s = scaler.fit_transform(X_train_b)
Xte_b_s = scaler.transform(X_test_b)

scaler2  = StandardScaler()
Xtr_e_s  = scaler2.fit_transform(X_train_e)
Xte_e_s  = scaler2.transform(X_test_e)

# ══════════════════════════════════════════════════════════════════
# 4. TRAIN MODELS
# ══════════════════════════════════════════════════════════════════
models = {
    'Linear Regression':      (LinearRegression(),            Xtr_b_s, Xte_b_s),
    'Ridge Regression':       (Ridge(alpha=1.0),              Xtr_b_s, Xte_b_s),
    'Lasso Regression':       (Lasso(alpha=0.1),              Xtr_b_s, Xte_b_s),
    'ElasticNet':             (ElasticNet(alpha=0.1,l1_ratio=0.5), Xtr_b_s, Xte_b_s),
    'Polynomial Regression':  (LinearRegression(),            None,    None),
    'Random Forest':          (RandomForestRegressor(n_estimators=100, random_state=42),
                               Xtr_e_s, Xte_e_s),
    'Gradient Boosting':      (GradientBoostingRegressor(n_estimators=100, random_state=42),
                               Xtr_e_s, Xte_e_s),
}

# Polynomial prep
poly   = PolynomialFeatures(degree=2, include_bias=False)
Xtr_p  = poly.fit_transform(X_train_b)
Xte_p  = poly.transform(X_test_b)
models['Polynomial Regression'] = (LinearRegression(), Xtr_p, Xte_p)

results = {}
print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)
for name, (model, Xtr, Xte) in models.items():
    model.fit(Xtr, y_train)
    y_pred  = model.predict(Xte)
    rmse    = np.sqrt(mean_squared_error(y_test, y_pred))
    mae     = mean_absolute_error(y_test, y_pred)
    r2      = r2_score(y_test, y_pred)
    results[name] = {'model': model, 'y_pred': y_pred,
                     'RMSE': round(rmse,3), 'MAE': round(mae,3), 'R2': round(r2,4)}
    print(f"\n{name}:  RMSE={rmse:.3f}  MAE={mae:.3f}  R²={r2:.4f}")

best_name = min(results, key=lambda k: results[k]['RMSE'])
print(f"\n✓ Best model: {best_name}  (RMSE={results[best_name]['RMSE']})")

# ══════════════════════════════════════════════════════════════════
# 5. ADVERTISING IMPACT ANALYSIS
# ══════════════════════════════════════════════════════════════════
lr = LinearRegression().fit(scaler.transform(X_base), y)
coef_df = pd.DataFrame({'Channel': features_base, 'Coefficient': lr.coef_})
print(f"\nLinear regression coefficients (standardised):\n{coef_df}")

rf_model = results['Random Forest']['model']
perm = permutation_importance(rf_model, Xte_e_s, y_test, n_repeats=10, random_state=42)
feat_imp = pd.DataFrame({'Feature': features_extended,
                         'Importance': perm.importances_mean}).sort_values('Importance', ascending=False)
print(f"\nPermutation importances (RF):\n{feat_imp.round(4)}")

# ROI: $ sales per $ ad spend (linear model, raw features)
lr_raw = LinearRegression().fit(X_train_b, y_train)
roi_df = pd.DataFrame({'Channel': features_base,
                       'Sales_per_$1000': lr_raw.coef_}).sort_values('Sales_per_$1000', ascending=False)
print(f"\nEstimated ROI (linear model):\n{roi_df.round(4)}")

# ══════════════════════════════════════════════════════════════════
# 6. SCENARIO FORECASTING
# ══════════════════════════════════════════════════════════════════
scenarios = pd.DataFrame({
    'Scenario':   ['Conservative', 'Balanced', 'TV-Heavy', 'Digital-Heavy', 'Aggressive'],
    'TV':         [50,  150, 250,  80, 300],
    'Radio':      [10,   30,  20,  60,  50],
    'Newspaper':  [10,   20,  15,  10,  30],
})
best_model  = results[best_name]['model']
best_scaler = scaler2 if best_name in ('Random Forest','Gradient Boosting') else scaler

if best_name in ('Random Forest', 'Gradient Boosting'):
    s_feats = scenarios[['TV','Radio','Newspaper']].copy()
    s_feats['TV_Radio_Interaction'] = s_feats['TV'] * s_feats['Radio'] / 1000
    s_feats['TV_Share']   = s_feats['TV']    / (s_feats['TV']+s_feats['Radio']+s_feats['Newspaper'])
    s_feats['Radio_Share']= s_feats['Radio'] / (s_feats['TV']+s_feats['Radio']+s_feats['Newspaper'])
    s_feats['Log_TV']     = np.log1p(s_feats['TV'])
    s_feats['Log_Radio']  = np.log1p(s_feats['Radio'])
    s_input = best_scaler.transform(s_feats)
elif best_name == 'Polynomial Regression':
    s_input = poly.transform(scaler.transform(scenarios[['TV','Radio','Newspaper']]))
else:
    s_input = best_scaler.transform(scenarios[['TV','Radio','Newspaper']])

scenarios['Predicted_Sales'] = best_model.predict(s_input).round(2)
print(f"\nScenario forecasts ({best_name}):\n{scenarios.to_string(index=False)}")

# ══════════════════════════════════════════════════════════════════
# 7. VISUALISATIONS
# ══════════════════════════════════════════════════════════════════
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#5DA5DA','#FAA43A','#60BD68','#F17CB0','#B276B2','#DECF3F','#F15854']

# ── Figure 1: EDA & Correlation ───────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Sales Prediction – Exploratory Data Analysis', fontsize=15, fontweight='bold', y=1.01)

channels = ['TV', 'Radio', 'Newspaper']
ch_colors = ['#5DA5DA','#FAA43A','#60BD68']
for i, (ch, col) in enumerate(zip(channels, ch_colors)):
    ax = axes[0, i]
    ax.scatter(df[ch], df['Sales'], alpha=0.6, color=col, edgecolors='white', linewidths=0.4, s=50)
    m, b = np.polyfit(df[ch], df['Sales'], 1)
    x_line = np.linspace(df[ch].min(), df[ch].max(), 200)
    ax.plot(x_line, m*x_line+b, color='#333', linewidth=1.5, linestyle='--')
    ax.set_xlabel(f'{ch} Spend ($000)', fontsize=10)
    ax.set_ylabel('Sales ($000)', fontsize=10)
    ax.set_title(f'{ch} vs Sales', fontsize=11, fontweight='bold')
    corr = df[ch].corr(df['Sales'])
    ax.text(0.05, 0.92, f'r = {corr:.3f}', transform=ax.transAxes,
            fontsize=10, color='#333', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

corr_matrix = df[['TV','Radio','Newspaper','Sales','Total_Spend','TV_Radio_Interaction']].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            ax=axes[1,0], linewidths=0.5, cbar_kws={'shrink':0.8}, annot_kws={'size':9})
axes[1,0].set_title('Correlation Heatmap', fontsize=11, fontweight='bold')
axes[1,0].tick_params(axis='x', rotation=30)

df['Budget_Segment_str'] = df['Budget_Segment'].astype(str)
seg_order = ['Low','Medium','High','Very High']
seg_means = df.groupby('Budget_Segment_str')['Sales'].mean().reindex(seg_order)
axes[1,1].bar(seg_order, seg_means, color=COLORS[:4], edgecolor='white', linewidth=0.5)
axes[1,1].set_xlabel('Budget Segment', fontsize=10)
axes[1,1].set_ylabel('Average Sales ($000)', fontsize=10)
axes[1,1].set_title('Avg Sales by Budget Segment', fontsize=11, fontweight='bold')
for i, v in enumerate(seg_means):
    axes[1,1].text(i, v+0.2, f'${v:.1f}k', ha='center', fontsize=9, fontweight='bold')

spend_share = df[['TV','Radio','Newspaper']].mean()
wedge_props = dict(width=0.5, edgecolor='white', linewidth=2)
axes[1,2].pie(spend_share, labels=['TV','Radio','Newspaper'], colors=ch_colors,
              autopct='%1.1f%%', startangle=90, wedgeprops=wedge_props,
              textprops={'fontsize':10})
axes[1,2].set_title('Avg Ad Spend Distribution', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: eda_analysis.png")

# ── Figure 2: Model Comparison ────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Sales Prediction – Model Evaluation', fontsize=15, fontweight='bold', y=1.01)

model_names  = list(results.keys())
rmse_vals    = [results[n]['RMSE'] for n in model_names]
r2_vals      = [results[n]['R2']   for n in model_names]
short_names  = ['LinReg','Ridge','Lasso','ElasticNet','Poly','RF','GBM']

bar_colors = ['#B276B2' if n==best_name else '#5DA5DA' for n in model_names]
bars = axes[0,0].bar(short_names, rmse_vals, color=bar_colors, edgecolor='white', linewidth=0.5)
axes[0,0].set_ylabel('RMSE (lower = better)', fontsize=10)
axes[0,0].set_title('RMSE by Model', fontsize=11, fontweight='bold')
axes[0,0].tick_params(axis='x', rotation=30)
for bar, v in zip(bars, rmse_vals):
    axes[0,0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                   f'{v:.2f}', ha='center', fontsize=8, fontweight='bold')

bar_colors2 = ['#B276B2' if n==best_name else '#60BD68' for n in model_names]
bars2 = axes[0,1].bar(short_names, r2_vals, color=bar_colors2, edgecolor='white', linewidth=0.5)
axes[0,1].set_ylabel('R² Score (higher = better)', fontsize=10)
axes[0,1].set_title('R² Score by Model', fontsize=11, fontweight='bold')
axes[0,1].set_ylim(min(r2_vals)-0.05, 1.0)
axes[0,1].tick_params(axis='x', rotation=30)
for bar, v in zip(bars2, r2_vals):
    axes[0,1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                   f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')

best_pred = results[best_name]['y_pred']
axes[0,2].scatter(y_test, best_pred, color='#5DA5DA', alpha=0.7, edgecolors='white', linewidths=0.4, s=60)
lims = [min(y_test.min(), best_pred.min())-1, max(y_test.max(), best_pred.max())+1]
axes[0,2].plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
axes[0,2].set_xlabel('Actual Sales', fontsize=10)
axes[0,2].set_ylabel('Predicted Sales', fontsize=10)
axes[0,2].set_title(f'Actual vs Predicted ({best_name})', fontsize=11, fontweight='bold')
axes[0,2].legend(fontsize=9)

residuals = y_test.values - best_pred
axes[1,0].scatter(best_pred, residuals, color='#FAA43A', alpha=0.7, edgecolors='white', linewidths=0.4, s=60)
axes[1,0].axhline(0, color='#333', linewidth=1.5, linestyle='--')
axes[1,0].set_xlabel('Predicted Sales', fontsize=10)
axes[1,0].set_ylabel('Residuals', fontsize=10)
axes[1,0].set_title('Residual Plot', fontsize=11, fontweight='bold')

axes[1,1].hist(residuals, bins=20, color='#60BD68', edgecolor='white', linewidth=0.5)
axes[1,1].set_xlabel('Residual Value', fontsize=10)
axes[1,1].set_ylabel('Frequency', fontsize=10)
axes[1,1].set_title('Residual Distribution', fontsize=11, fontweight='bold')
axes[1,1].axvline(0, color='#333', linewidth=1.5, linestyle='--')

top_feats = feat_imp.head(6)
axes[1,2].barh(top_feats['Feature'], top_feats['Importance'], color='#B276B2', edgecolor='white', linewidth=0.5)
axes[1,2].set_xlabel('Permutation Importance', fontsize=10)
axes[1,2].set_title('Top Feature Importances (RF)', fontsize=11, fontweight='bold')
axes[1,2].invert_yaxis()

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: model_evaluation.png")

# ── Figure 3: Business Insights ───────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Sales Prediction – Business Insights & Forecasting', fontsize=15, fontweight='bold', y=1.01)

tv_range = np.linspace(0, 300, 100)
fixed_radio, fixed_news = df['Radio'].mean(), df['Newspaper'].mean()
sim_data = pd.DataFrame({'TV': tv_range, 'Radio': fixed_radio, 'Newspaper': fixed_news})
sim_scaled = scaler.transform(sim_data)
axes[0,0].plot(tv_range, LinearRegression().fit(Xtr_b_s, y_train).predict(sim_scaled),
               color='#5DA5DA', linewidth=2.5, label='Predicted Sales')
axes[0,0].fill_between(tv_range,
    LinearRegression().fit(Xtr_b_s, y_train).predict(sim_scaled) - 1.5,
    LinearRegression().fit(Xtr_b_s, y_train).predict(sim_scaled) + 1.5,
    alpha=0.2, color='#5DA5DA', label='±1.5 confidence band')
axes[0,0].axvline(df['TV'].mean(), color='#FAA43A', linewidth=1.5, linestyle='--',
                  label=f"Mean TV (${df['TV'].mean():.0f}k)")
axes[0,0].set_xlabel('TV Spend ($000)', fontsize=10)
axes[0,0].set_ylabel('Predicted Sales ($000)', fontsize=10)
axes[0,0].set_title('TV Spend vs Predicted Sales\n(Radio & Newspaper held constant)', fontsize=11, fontweight='bold')
axes[0,0].legend(fontsize=9)

radio_range = np.linspace(0, 50, 100)
sim_radio = pd.DataFrame({'TV': df['TV'].mean(), 'Radio': radio_range, 'Newspaper': fixed_news})
sim_radio_scaled = scaler.transform(sim_radio)
lr_fit = LinearRegression().fit(Xtr_b_s, y_train)
axes[0,1].plot(radio_range, lr_fit.predict(sim_radio_scaled),
               color='#FAA43A', linewidth=2.5, label='Predicted Sales')
axes[0,1].axvline(df['Radio'].mean(), color='#5DA5DA', linewidth=1.5, linestyle='--',
                  label=f"Mean Radio (${df['Radio'].mean():.0f}k)")
axes[0,1].set_xlabel('Radio Spend ($000)', fontsize=10)
axes[0,1].set_ylabel('Predicted Sales ($000)', fontsize=10)
axes[0,1].set_title('Radio Spend vs Predicted Sales\n(TV & Newspaper held constant)', fontsize=11, fontweight='bold')
axes[0,1].legend(fontsize=9)

scen_colors = COLORS[:len(scenarios)]
bars = axes[1,0].bar(scenarios['Scenario'], scenarios['Predicted_Sales'],
                     color=scen_colors, edgecolor='white', linewidth=0.5)
axes[1,0].set_ylabel('Predicted Sales ($000)', fontsize=10)
axes[1,0].set_title(f'Scenario Forecast – {best_name}', fontsize=11, fontweight='bold')
axes[1,0].tick_params(axis='x', rotation=20)
for bar, v in zip(bars, scenarios['Predicted_Sales']):
    axes[1,0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                   f'${v:.1f}k', ha='center', fontsize=9, fontweight='bold')

roi_vals = lr_raw.coef_
ch_colors2 = ['#5DA5DA','#FAA43A','#60BD68']
roi_bars = axes[1,1].bar(features_base, roi_vals, color=ch_colors2, edgecolor='white', linewidth=0.5)
axes[1,1].set_ylabel('Sales Units per $1000 Spent', fontsize=10)
axes[1,1].set_title('Channel ROI (Linear Model)', fontsize=11, fontweight='bold')
for bar, v in zip(roi_bars, roi_vals):
    axes[1,1].text(bar.get_x()+bar.get_width()/2, max(v+0.003, 0.01),
                   f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('business_insights.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: business_insights.png")

# ══════════════════════════════════════════════════════════════════
# 8. SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ACTIONABLE BUSINESS INSIGHTS")
print("=" * 60)
top_channel = roi_df.iloc[0]
print(f"1. Highest ROI channel : {top_channel['Channel']} "
      f"(+{top_channel['Sales_per_$1000']:.4f} sales units per $1k spend)")
print(f"2. Best predictive model: {best_name} (R²={results[best_name]['R2']}, RMSE={results[best_name]['RMSE']})")
print(f"3. TV-Radio interaction is a strong combined driver (r²={df['TV_Radio_Interaction'].corr(df['Sales']):.3f} with Sales)")
print(f"4. Newspaper shows lowest ROI — consider reallocating budget to TV/Radio")
print(f"5. Best aggressive scenario forecast: ${scenarios['Predicted_Sales'].max():.2f}k in predicted sales")
