"""
=====================================================================
  MINI PROJECT: Does Money Buy Happiness?
  Data Visualisation of the World Happiness Report using Python
=====================================================================

All charts are also saved into a folder called "charts".
"""

# %% ---------------------------------------------------------------
# STEP 0: Import libraries
# ------------------------------------------------------------------
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid")


# %% ---------------------------------------------------------------
# STEP 1: Load the data
# ------------------------------------------------------------------
def read_csv_safely(path):
    """Some Kaggle CSVs are not UTF-8, so try a few encodings."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read {path}")


# Look in the script's own folder AND all its subfolders, so it works
# no matter where you run it from or if Kaggle unzipped into a subfolder.
try:
    BASE = os.path.dirname(os.path.abspath(__file__))
except NameError:                      # running inside Jupyter
    BASE = os.getcwd()
os.chdir(BASE)                         # charts folder is created next to script
os.makedirs("charts", exist_ok=True)

ALL_CSVS = glob.glob(os.path.join(BASE, "**", "*.csv"), recursive=True)


def find_file(must_have, must_not_have=None):
    """Find a CSV whose file name contains `must_have`."""
    for path in ALL_CSVS:
        name = os.path.basename(path).lower()
        if must_have in name and (must_not_have is None
                                  or must_not_have not in name):
            print("Using:", path)
            return path
    zips = glob.glob(os.path.join(BASE, "**", "*.zip"), recursive=True)
    msg = [f"\nCould not find a CSV with '{must_have}' in its name.",
           f"Searched folder: {BASE}",
           f"CSV files found: {[os.path.basename(p) for p in ALL_CSVS] or 'NONE'}"]
    if zips:
        msg.append(f"Zip files found: {[os.path.basename(z) for z in zips]}"
                   " -> double-click the zip to unzip it first!")
    raise FileNotFoundError("\n".join(msg))


ranking = read_csv_safely(find_file("2024", must_not_have="updated"))  # 2024 snapshot
yearly = read_csv_safely(find_file("updated"))                         # 2005-2023 panel

# Make column names consistent (strip spaces)
ranking.columns = ranking.columns.str.strip()
yearly.columns = yearly.columns.str.strip()

# Rename long column names to short, easy-to-explain names
yearly = yearly.rename(columns={
    "Country name": "Country",
    "year": "Year",
    "Life Ladder": "Happiness",
    "Log GDP per capita": "GDP",
    "Social support": "Social Support",
    "Healthy life expectancy at birth": "Life Expectancy",
    "Freedom to make life choices": "Freedom",
    "Perceptions of corruption": "Corruption",
})
ranking = ranking.rename(columns={
    "Country name": "Country",
    "Regional indicator": "Region",
    "Ladder score": "Happiness",
})

print("2024 ranking shape :", ranking.shape)
print("Yearly data shape  :", yearly.shape)
print(yearly.head())


# %% ---------------------------------------------------------------
# STEP 2: Clean the data
# ------------------------------------------------------------------
FACTORS = ["GDP", "Social Support", "Life Expectancy",
           "Freedom", "Generosity", "Corruption"]

print("\nMissing values per column (yearly data):")
print(yearly[["Happiness"] + FACTORS].isna().sum())

# Fill small gaps with each country's own average (better than dropping)
for col in FACTORS:
    yearly[col] = yearly.groupby("Country")[col].transform(
        lambda s: s.fillna(s.mean()))
yearly = yearly.dropna(subset=["Happiness"] + FACTORS)

# Add the Region column to the yearly data using the 2024 file
yearly = yearly.merge(ranking[["Country", "Region"]], on="Country", how="left")
yearly["Region"] = yearly["Region"].fillna("Other")

LATEST = int(yearly["Year"].max())
latest = yearly[yearly["Year"] == LATEST]
print(f"\nLatest year in yearly data: {LATEST} ({len(latest)} countries)")


# %% ---------------------------------------------------------------
# CHART 1: Top 10 happiest and 10 least happy countries (2024)
# ------------------------------------------------------------------
rank_sorted = ranking.dropna(subset=["Happiness"]).sort_values(
    "Happiness", ascending=False)
top10 = rank_sorted.head(10)
bottom10 = rank_sorted.tail(10)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.barplot(data=top10, x="Happiness", y="Country", ax=axes[0],
            hue="Country", palette="Greens_r", legend=False)
axes[0].set_title("Top 10 Happiest Countries (2024)", fontsize=14)
sns.barplot(data=bottom10, x="Happiness", y="Country", ax=axes[1],
            hue="Country", palette="Reds", legend=False)
axes[1].set_title("10 Least Happy Countries (2024)", fontsize=14)
for ax in axes:
    ax.set_xlim(0, 8)
    ax.set_xlabel("Happiness Score (0-10)")
plt.tight_layout()
plt.savefig("charts/1_top_bottom_10.png", dpi=150)
plt.show()


# %% ---------------------------------------------------------------
# CHART 2: World map of happiness (interactive)
# ------------------------------------------------------------------
fig = px.choropleth(
    ranking, locations="Country", locationmode="country names",
    color="Happiness", hover_name="Country",
    color_continuous_scale="RdYlGn",
    title="World Happiness Map (2024)")
fig.write_html("charts/2_world_map.html")
fig.show()


# %% ---------------------------------------------------------------
# CHART 3: Does money buy happiness? GDP vs Happiness
# ------------------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.regplot(data=latest, x="GDP", y="Happiness",
            scatter_kws={"alpha": 0.6}, line_kws={"color": "red"})
india = latest[latest["Country"] == "India"]
if not india.empty:
    plt.scatter(india["GDP"], india["Happiness"], color="orange",
                s=200, edgecolor="black", zorder=5, label="India")
    plt.legend()
corr = latest["GDP"].corr(latest["Happiness"])
plt.title(f"Does Money Buy Happiness? ({LATEST})  |  correlation = {corr:.2f}",
          fontsize=14)
plt.xlabel("Log GDP per Capita (richer ->)")
plt.ylabel("Happiness Score")
plt.tight_layout()
plt.savefig("charts/3_gdp_vs_happiness.png", dpi=150)
plt.show()


# %% ---------------------------------------------------------------
# CHART 4: Correlation heatmap - which factors move with happiness?
# ------------------------------------------------------------------
plt.figure(figsize=(9, 7))
corr_matrix = latest[["Happiness"] + FACTORS].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, square=True)
plt.title(f"How Factors Relate to Happiness ({LATEST})", fontsize=14)
plt.tight_layout()
plt.savefig("charts/4_correlation_heatmap.png", dpi=150)
plt.show()


# %% ---------------------------------------------------------------
# CHART 5: Happiness by region (box plot)
# ------------------------------------------------------------------
region_data = ranking.dropna(subset=["Region", "Happiness"])
order = (region_data.groupby("Region")["Happiness"].median()
         .sort_values(ascending=False).index)

plt.figure(figsize=(12, 7))
sns.boxplot(data=region_data, y="Region", x="Happiness", order=order,
            hue="Region", palette="viridis", legend=False)
plt.title("Happiness Across World Regions (2024)", fontsize=14)
plt.xlabel("Happiness Score")
plt.ylabel("")
plt.tight_layout()
plt.savefig("charts/5_region_boxplot.png", dpi=150)
plt.show()


# %% ---------------------------------------------------------------
# CHART 6: Animated bubble chart over the years (Hans Rosling style)
# ------------------------------------------------------------------
anim = yearly[yearly["Year"] >= 2010].sort_values("Year")
fig = px.scatter(
    anim, x="GDP", y="Happiness", animation_frame="Year",
    animation_group="Country", size="Life Expectancy", color="Region",
    hover_name="Country", size_max=25,
    range_x=[anim["GDP"].min() - 0.3, anim["GDP"].max() + 0.3],
    range_y=[1.5, 8.5],
    title="Wealth vs Happiness Over the Years (press Play)")
fig.write_html("charts/6_animated_bubbles.html")
fig.show()


# %% ---------------------------------------------------------------
# CHART 7: India's happiness trend vs the world average
# ------------------------------------------------------------------
world_avg = yearly.groupby("Year")["Happiness"].mean()
india_trend = yearly[yearly["Country"] == "India"].set_index("Year")["Happiness"]

plt.figure(figsize=(11, 6))
plt.plot(world_avg.index, world_avg.values, marker="o",
         label="World Average", linewidth=2)
plt.plot(india_trend.index, india_trend.values, marker="o",
         label="India", linewidth=2, color="orange")
plt.title("India vs World: Happiness Over Time", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Happiness Score")
plt.legend()
plt.tight_layout()
plt.savefig("charts/7_india_trend.png", dpi=150)
plt.show()


# %% ---------------------------------------------------------------
# CHART 8: Radar chart - India vs neighbours on each factor
# ------------------------------------------------------------------
neighbours = ["India", "China", "Pakistan", "Bangladesh", "Nepal", "Sri Lanka"]

# Scale every factor to 0-1 so they can share one chart
scaled = latest.copy()
for col in FACTORS:
    lo, hi = latest[col].min(), latest[col].max()
    scaled[col] = (latest[col] - lo) / (hi - lo)
# For corruption, LOWER is better, so flip it into "Trust"
scaled["Corruption"] = 1 - scaled["Corruption"]
radar_labels = ["GDP", "Social Support", "Life Expectancy",
                "Freedom", "Generosity", "Trust (low corruption)"]

fig = go.Figure()
for country in neighbours:
    row = scaled[scaled["Country"] == country]
    if row.empty:
        continue
    values = row[FACTORS].values.flatten().tolist()
    fig.add_trace(go.Scatterpolar(
        r=values + values[:1], theta=radar_labels + radar_labels[:1],
        fill="toself", name=country, opacity=0.6))
fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    title=f"India vs Neighbours: Strengths & Weaknesses ({LATEST})")
fig.write_html("charts/8_radar_neighbours.html")
fig.show()


# %% ---------------------------------------------------------------
# CHART 9: Simple prediction - which factor matters MOST?
# ------------------------------------------------------------------
X = latest[FACTORS]
y = latest["Happiness"]
X_scaled = StandardScaler().fit_transform(X)   # put factors on same scale
model = LinearRegression().fit(X_scaled, y)
r2 = model.score(X_scaled, y)

importance = (pd.Series(model.coef_, index=FACTORS)
              .sort_values(key=abs, ascending=True))

plt.figure(figsize=(10, 6))
colors = ["green" if v > 0 else "red" for v in importance]
importance.plot(kind="barh", color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title(f"What Predicts Happiness? (Linear Regression, R² = {r2:.2f})",
          fontsize=14)
plt.xlabel("Effect on happiness (green = increases, red = decreases)")
plt.tight_layout()
plt.savefig("charts/9_factor_importance.png", dpi=150)
plt.show()


# %% ---------------------------------------------------------------
# STEP 10: Key insights (printed automatically from the data)
# ------------------------------------------------------------------
happiest = rank_sorted.iloc[0]
saddest = rank_sorted.iloc[-1]
best_region = order[0]
top_factor = importance.abs().idxmax()

print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
print(f"1. Happiest country in 2024: {happiest['Country']} "
      f"({happiest['Happiness']:.2f}); least happy: {saddest['Country']} "
      f"({saddest['Happiness']:.2f}).")
print(f"2. Happiest region overall: {best_region}.")
print(f"3. GDP and happiness have a correlation of {corr:.2f} - money helps,"
      " but it is not the whole story.")
print(f"4. The single strongest predictor of happiness is '{top_factor}'. "
      f"Together the six factors explain {r2 * 100:.0f}% of the differences "
      "between countries.")
india_rank = rank_sorted.reset_index(drop=True)
india_pos = india_rank.index[india_rank["Country"] == "India"]
if len(india_pos):
    india_score = india_rank.loc[india_pos[0], "Happiness"]
    world_mean = india_rank["Happiness"].mean()
    side = "below" if india_score < world_mean else "above"
    print(f"5. India ranks #{india_pos[0] + 1} of {len(india_rank)} in 2024 "
          f"with {india_score:.2f}, {side} the world average of "
          f"{world_mean:.2f}.")
print("\nAll charts saved in the 'charts' folder.")