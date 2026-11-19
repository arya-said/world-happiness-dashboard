"""
WORLD HAPPINESS DASHBOARD  (Streamlit frontend for the project)

Puts every chart on ONE web page with tabs. A country picker at the
top lets you highlight any country (default: India).

RUN: streamlit run app.py

DATABASE: on the first run the CSV data is copied into an SQLite
database (happiness.db). After that, every chart reads from the
database.
"""

import os
import glob
import sqlite3
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import db   # our database layer (db.py)

st.set_page_config(page_title="World Happiness Dashboard",
                   page_icon="🌍", layout="wide")

FACTORS = ["GDP", "Social Support", "Life Expectancy",
           "Freedom", "Generosity", "Corruption"]


# ------------------------------------------------------------------
# 1. LOAD + CLEAN DATA (cached so the page stays fast)
# ------------------------------------------------------------------
def read_csv_safely(path):
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read {path}")


def find_file(all_csvs, must_have, must_not_have=None):
    for path in all_csvs:
        name = os.path.basename(path).lower()
        if must_have in name and (must_not_have is None
                                  or must_not_have not in name):
            return path
    return None


def import_from_csv():
    """Copy the Kaggle CSV files into the database. Returns False if not found."""
    base = os.path.dirname(os.path.abspath(__file__))
    csvs = glob.glob(os.path.join(base, "**", "*.csv"), recursive=True)
    rank_path = find_file(csvs, "2024", must_not_have="updated")
    year_path = find_file(csvs, "updated")
    if rank_path is None or year_path is None:
        return False
    db.import_csvs(read_csv_safely(rank_path), read_csv_safely(year_path))
    return True


@st.cache_data
def load_data():
    # First run: the database is empty, so fill it from the CSV files
    if db.is_empty() and not import_from_csv():
        return None, None

    # Every chart reads from the DATABASE (not the CSV files)
    ranking = db.fetch_ranking()
    yearly = db.fetch_yearly()

    yearly = yearly.rename(columns={
        "country": "Country", "year": "Year", "happiness": "Happiness",
        "gdp": "GDP", "social_support": "Social Support",
        "life_expectancy": "Life Expectancy", "freedom": "Freedom",
        "generosity": "Generosity", "corruption": "Corruption"})
    ranking = ranking.rename(columns={
        "country": "Country", "region": "Region", "happiness": "Happiness"})

    # Fill gaps with each country's own average. If a country has NO data
    # for a factor at all (e.g. China never reports corruption), use that
    # year's world median instead, so the country isn't dropped.
    for col in FACTORS:
        yearly[col] = yearly.groupby("Country")[col].transform(
            lambda s: s.fillna(s.mean()))
        yearly[col] = yearly.groupby("Year")[col].transform(
            lambda s: s.fillna(s.median()))
    yearly = yearly.dropna(subset=["Happiness"] + FACTORS)
    yearly = yearly.merge(ranking[["Country", "Region"]],
                          on="Country", how="left")
    yearly["Region"] = yearly["Region"].fillna("Other")

    ranking = (ranking.dropna(subset=["Happiness"])
               .sort_values("Happiness", ascending=False)
               .reset_index(drop=True))
    ranking["Rank"] = ranking.index + 1
    return ranking, yearly


ranking, yearly = load_data()
if ranking is None:
    st.error("Could not find the Kaggle CSV files. Put the 'archive' "
             "folder (or both CSVs) next to app.py and refresh.")
    st.stop()

# Latest year with enough countries (so one inserted 2024 row can't
# turn "latest year" into a year with a single country)
year_counts = yearly.groupby("Year")["Country"].count()
enough = year_counts[year_counts >= min(50, year_counts.max())]
LATEST = int(enough.index.max())
latest = yearly[yearly["Year"] == LATEST]


# ------------------------------------------------------------------
# 2. HEADER + COUNTRY PICKER (no sidebar)
# ------------------------------------------------------------------
st.title("Does Money Buy Happiness?")
st.markdown("An interactive look at the **World Happiness Report**: "
            "which countries are happiest, and what really drives it.")
st.caption("Data: World Happiness Report (Gallup World Poll), via Kaggle. "
           "Scores are on a 0-10 scale.")

countries = sorted(ranking["Country"].unique())
pick_col, _ = st.columns([1, 3])
focus = pick_col.selectbox(
    "🌍 Country to highlight", countries,
    index=countries.index("India") if "India" in countries else 0,
    help="Changes the rank card below, the orange dot on the GDP chart, "
         "the Focus tab and the Insights tab.")

# Regression (used in the KPI row and the 'What matters' tab)
X_scaled = StandardScaler().fit_transform(latest[FACTORS])
model = LinearRegression().fit(X_scaled, latest["Happiness"])
r2 = model.score(X_scaled, latest["Happiness"])
importance = pd.Series(model.coef_, index=FACTORS).sort_values(key=abs)
gdp_corr = latest["GDP"].corr(latest["Happiness"])

focus_row = ranking[ranking["Country"] == focus].iloc[0]
world_mean = ranking["Happiness"].mean()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Happiest country (2024)", ranking.iloc[0]["Country"],
          f"{ranking.iloc[0]['Happiness']:.2f} / 10", delta_color="off")
k2.metric(f"{focus}'s rank (2024)",
          f"#{int(focus_row['Rank'])} of {len(ranking)}",
          f"{focus_row['Happiness'] - world_mean:+.2f} vs world avg")
k3.metric("GDP ↔ Happiness correlation", f"{gdp_corr:.2f}")
k4.metric("Explained by 6 factors", f"{r2 * 100:.0f}%")

st.markdown("")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏆 Rankings", "💰 Money & Happiness", "🗺️ Regions",
    f"🇮🇳 {focus} Focus" if focus == "India" else f"📍 {focus} Focus",
    "🔍 What Matters Most", "💡 Insights", "🗄️ Manage Data"])


# ------------------------------------------------------------------
# TAB 1: Rankings + world map
# ------------------------------------------------------------------
with tab1:
    c1, c2 = st.columns(2)
    top10 = ranking.head(10).sort_values("Happiness")
    bottom10 = ranking.tail(10).sort_values("Happiness", ascending=False)

    fig = px.bar(top10, x="Happiness", y="Country", orientation="h",
                 color="Happiness", color_continuous_scale="Greens",
                 range_x=[0, 8], title="Top 10 Happiest Countries")
    fig.update_layout(coloraxis_showscale=False, height=420)
    c1.plotly_chart(fig)

    fig = px.bar(bottom10, x="Happiness", y="Country", orientation="h",
                 color="Happiness", color_continuous_scale="Reds_r",
                 range_x=[0, 8], title="10 Least Happy Countries")
    fig.update_layout(coloraxis_showscale=False, height=420)
    c2.plotly_chart(fig)

    fig = px.choropleth(ranking, locations="Country",
                        locationmode="country names", color="Happiness",
                        hover_name="Country", hover_data=["Rank"],
                        color_continuous_scale="RdYlGn",
                        title="World Happiness Map (2024)")
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig)


# ------------------------------------------------------------------
# TAB 2: GDP vs happiness, heatmap, animation
# ------------------------------------------------------------------
with tab2:
    c1, c2 = st.columns([3, 2])
    fig = px.scatter(latest, x="GDP", y="Happiness", hover_name="Country",
                     color="Region", opacity=0.7,
                     title=f"Richer countries tend to be happier "
                           f"({LATEST}, correlation = {gdp_corr:.2f})",
                     labels={"GDP": "Log GDP per capita (richer →)"})
    # trend line drawn with numpy (no extra library needed)
    m, b = np.polyfit(latest["GDP"], latest["Happiness"], 1)
    xs = np.linspace(latest["GDP"].min(), latest["GDP"].max(), 50)
    fig.add_trace(go.Scatter(x=xs, y=m * xs + b, mode="lines",
                             name="Trend", line=dict(color="black",
                                                     dash="dash")))
    f = latest[latest["Country"] == focus]
    if not f.empty:
        fig.add_trace(go.Scatter(
            x=f["GDP"], y=f["Happiness"], mode="markers+text",
            text=[focus], textposition="top center", name=focus,
            marker=dict(size=18, color="orange",
                        line=dict(width=2, color="black"))))
    fig.update_layout(height=480)
    c1.plotly_chart(fig)

    corr = latest[["Happiness"] + FACTORS].corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, title="How factors relate")
    fig.update_layout(height=480)
    c2.plotly_chart(fig)

    st.subheader("Wealth vs Happiness over time")
    st.caption("Press ▶ Play. Bubble size = life expectancy.")
    anim = yearly[yearly["Year"] >= 2010].sort_values("Year")
    fig = px.scatter(anim, x="GDP", y="Happiness", animation_frame="Year",
                     animation_group="Country", size="Life Expectancy",
                     color="Region", hover_name="Country", size_max=22,
                     range_x=[anim["GDP"].min() - 0.3,
                              anim["GDP"].max() + 0.3],
                     range_y=[1.5, 8.5])
    fig.update_layout(height=560)
    st.plotly_chart(fig)


# ------------------------------------------------------------------
# TAB 3: Regions
# ------------------------------------------------------------------
with tab3:
    reg = ranking.dropna(subset=["Region"])
    order = (reg.groupby("Region")["Happiness"].median()
             .sort_values().index.tolist())
    fig = px.box(reg, x="Happiness", y="Region", color="Region",
                 points="all", hover_name="Country",
                 category_orders={"Region": order[::-1]},
                 title="Happiness across world regions (2024)")
    fig.update_layout(showlegend=False, height=600)
    st.plotly_chart(fig)

    avg = (reg.groupby("Region")[["Happiness"]].mean().round(2)
           .sort_values("Happiness", ascending=False))
    avg["Countries"] = reg.groupby("Region").size()
    st.dataframe(avg)


# ------------------------------------------------------------------
# TAB 4: Focus country - trend + radar
# ------------------------------------------------------------------
with tab4:
    # Only use years where enough countries were surveyed
    counts = yearly.groupby("Year")["Country"].count()
    good_years = counts[counts >= 50].index
    world_avg = (yearly[yearly["Year"].isin(good_years)]
                 .groupby("Year")["Happiness"].mean().reset_index())
    world_avg["Series"] = "World average"
    trend = yearly[yearly["Country"] == focus][["Year", "Happiness"]].copy()
    trend["Series"] = focus
    both = pd.concat([world_avg, trend])

    fig = px.line(both, x="Year", y="Happiness", color="Series",
                  markers=True, title=f"{focus} vs World: happiness over time",
                  color_discrete_map={"World average": "#4C72B0",
                                      focus: "orange"})
    fig.update_xaxes(dtick=1)
    fig.update_layout(height=450)
    st.plotly_chart(fig)

    # Radar: each country's most recent year, scaled 0-1 across countries
    recent = yearly.sort_values("Year").groupby("Country").tail(1)
    scaled = recent.copy()
    for col in FACTORS:
        lo, hi = recent[col].min(), recent[col].max()
        scaled[col] = (recent[col] - lo) / (hi - lo)
    scaled["Corruption"] = 1 - scaled["Corruption"]   # higher = more trust
    labels = ["GDP", "Social Support", "Life Expectancy",
              "Freedom", "Generosity", "Trust (low corruption)"]

    st.subheader(f"{focus} vs selected countries")
    default_compare = [c for c in ["China", "Pakistan", "Bangladesh",
                                   "Nepal", "Sri Lanka"]
                       if c in countries and c != focus]
    compare = st.multiselect(
        "Compare with", [c for c in countries if c != focus],
        default=default_compare,
        help="Add or remove countries to show on the radar chart.")
    fig = go.Figure()
    missing = []
    for c in [focus] + compare:
        row = scaled[scaled["Country"] == c]
        if row.empty:
            missing.append(c)
            continue
        vals = row[FACTORS].values.flatten().tolist()
        fig.add_trace(go.Scatterpolar(
            r=vals + vals[:1], theta=labels + labels[:1], fill="toself",
            name=c, opacity=0.55,
            line=dict(width=4 if c == focus else 1.5)))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1])),
                      title="Strengths & weaknesses "
                            "(0 = worst country, 1 = best country)",
                      height=560)
    st.plotly_chart(fig)
    if missing:
        st.warning("No yearly data available for: " + ", ".join(missing))


# ------------------------------------------------------------------
# TAB 5: Regression - what matters most
# ------------------------------------------------------------------
with tab5:
    imp = importance.reset_index()
    imp.columns = ["Factor", "Effect"]
    imp["Direction"] = imp["Effect"].apply(
        lambda v: "Increases happiness" if v > 0 else "Decreases happiness")
    fig = px.bar(imp, x="Effect", y="Factor", orientation="h",
                 color="Direction",
                 color_discrete_map={"Increases happiness": "#2E8B57",
                                     "Decreases happiness": "#D9534F"},
                 title=f"What predicts happiness? "
                       f"(Linear regression, R² = {r2:.2f})")
    fig.update_layout(height=450)
    st.plotly_chart(fig)
    st.info(f"**How to read this:** factors were put on the same scale, so "
            f"longer bars mean a bigger effect. Together they explain "
            f"**{r2 * 100:.0f}%** of the differences in happiness between "
            f"countries in {LATEST}.")


# ------------------------------------------------------------------
# TAB 6: Insights (all computed from the data)
# ------------------------------------------------------------------
with tab6:
    top_factor = importance.abs().idxmax()
    best_region = (ranking.groupby("Region")["Happiness"].median()
                   .idxmax())
    side = "below" if focus_row["Happiness"] < world_mean else "above"
    st.markdown(f"""
1. **{ranking.iloc[0]['Country']}** is the happiest country in 2024
   ({ranking.iloc[0]['Happiness']:.2f}), while **{ranking.iloc[-1]['Country']}**
   is the least happy ({ranking.iloc[-1]['Happiness']:.2f}).
2. **{best_region}** is the happiest region overall.
3. Money matters: GDP and happiness have a correlation of **{gdp_corr:.2f}**,
   but it is not the whole story.
4. The strongest single predictor is **{top_factor}**. The six factors together
   explain **{r2 * 100:.0f}%** of the differences between countries.
5. **{focus}** ranks **#{int(focus_row['Rank'])} of {len(ranking)}**
   ({focus_row['Happiness']:.2f}), {side} the world average of
   {world_mean:.2f}.
""")
    with st.expander("See the raw 2024 ranking data"):
        st.dataframe(ranking)


# ------------------------------------------------------------------
# TAB 7: Manage Data  (database INSERT / UPDATE / DELETE)
# ------------------------------------------------------------------
def refresh(message):
    """After a change: forget the cached data so every chart re-reads the database."""
    load_data.clear()
    st.session_state["flash"] = message
    st.rerun()


def num(value, default):
    """Database values can be empty (NULL); use a default for the input box."""
    return float(value) if pd.notna(value) else default


with tab7:
    st.subheader("Manage the database")
    st.caption("All charts read from the SQLite database file **happiness.db**. "
               "Any change made here updates every chart on the page.")
    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))

    with st.expander("Which charts does each table feed?", expanded=False):
        st.markdown(
            "| Table | Changes show up in |\n|---|---|\n"
            "| **yearly** (country + year rows) | Focus tab: trend line and radar; "
            "animated bubble chart (years up to 2023) |\n"
            "| **ranking** (2024 scores) | Rank card at the top, Rankings tab, world map, "
            "Regions tab, Insights |")

    raw = db.fetch_yearly()              # rows exactly as stored (before cleaning)
    raw_rank = db.fetch_ranking()
    m1, m2, m3 = st.columns(3)
    m1.metric("Countries in 2024 ranking", len(raw_rank))
    m2.metric("Yearly records", f"{len(raw):,}")
    m3.metric("Years covered", f"{raw['year'].min()} – {raw['year'].max()}")

    action = st.radio("Choose an action",
                      ["View", "Insert", "Update", "Delete", "Activity log", "Reset"],
                      horizontal=True)
    yearly_countries = sorted(raw["country"].unique())

    # ---------- VIEW (SELECT) ----------
    if action == "View":
        pick = st.selectbox("Show records for", ["All countries"] + yearly_countries,
                            key="view_country")
        shown = raw if pick == "All countries" else raw[raw["country"] == pick]
        st.dataframe(shown, hide_index=True)
        st.caption(f"{len(shown)} rows  ·  SQL: SELECT * FROM yearly"
                   + ("" if pick == "All countries" else f" WHERE country = '{pick}'"))

    # ---------- INSERT ----------
    elif action == "Insert":
        st.markdown("**Add a yearly record** (one country, one year)")
        with st.form("insert_yearly", clear_on_submit=True):
            a, b = st.columns(2)
            country = a.text_input("Country", value=focus,
                                   help="An existing country, or a new name")
            year = b.number_input("Year", min_value=2005, max_value=2030, value=2024, step=1)
            a, b, c, d = st.columns(4)
            happiness = a.number_input("Happiness (0-10)", 0.0, 10.0, 5.0, 0.01)
            gdp = b.number_input("Log GDP per capita", 5.0, 12.5, 9.0, 0.01)
            social = c.number_input("Social support (0-1)", 0.0, 1.0, 0.8, 0.01)
            life = d.number_input("Life expectancy (years)", 30.0, 85.0, 65.0, 0.1)
            a, b, c = st.columns(3)
            freedom = a.number_input("Freedom (0-1)", 0.0, 1.0, 0.75, 0.01)
            generosity = b.number_input("Generosity", -0.5, 1.0, 0.0, 0.01)
            corruption = c.number_input("Corruption (0-1)", 0.0, 1.0, 0.75, 0.01)
            if st.form_submit_button("Insert record", type="primary"):
                if not country.strip():
                    st.error("Please enter a country name.")
                else:
                    try:
                        db.insert_yearly({
                            "country": country.strip(), "year": int(year),
                            "happiness": happiness, "gdp": gdp,
                            "social_support": social, "life_expectancy": life,
                            "freedom": freedom, "generosity": generosity,
                            "corruption": corruption})
                        refresh(f"Inserted {country.strip()} {int(year)}. "
                                f"Pick {country.strip()} at the top and open its Focus tab to see the new point.")
                    except sqlite3.IntegrityError:
                        st.error(f"{country.strip()} {int(year)} already exists. "
                                 "Use Update to change it.")

        st.markdown("**Add a country to the 2024 ranking**")
        with st.form("insert_ranking", clear_on_submit=True):
            a, b, c = st.columns(3)
            new_country = a.text_input("Country name")
            region = b.selectbox("Region", sorted(raw_rank["region"].dropna().unique()))
            score = c.number_input("2024 happiness (0-10)", 0.0, 10.0, 5.0, 0.01)
            if st.form_submit_button("Add to ranking", type="primary"):
                if not new_country.strip():
                    st.error("Please enter a country name.")
                else:
                    try:
                        db.insert_ranking(new_country.strip(), region, score)
                        refresh(f"Added {new_country.strip()} to the 2024 ranking. "
                                f"It now appears in the Rankings tab, the map and the country list.")
                    except sqlite3.IntegrityError:
                        st.error(f"{new_country.strip()} is already in the ranking.")

    # ---------- UPDATE ----------
    elif action == "Update":
        st.markdown("**Change a yearly record**")
        a, b = st.columns(2)
        u_country = a.selectbox("Country", yearly_countries,
                                index=yearly_countries.index(focus) if focus in yearly_countries else 0,
                                key="upd_country")
        years = sorted(raw[raw["country"] == u_country]["year"], reverse=True)
        u_year = b.selectbox("Year", years, key="upd_year")
        row = raw[(raw["country"] == u_country) & (raw["year"] == u_year)].iloc[0]

        with st.form("update_yearly"):
            a, b, c, d = st.columns(4)
            happiness = a.number_input("Happiness (0-10)", 0.0, 10.0, num(row["happiness"], 5.0), 0.01)
            gdp = b.number_input("Log GDP per capita", 5.0, 12.5, num(row["gdp"], 9.0), 0.01)
            social = c.number_input("Social support (0-1)", 0.0, 1.0, num(row["social_support"], 0.8), 0.01)
            life = d.number_input("Life expectancy (years)", 30.0, 85.0, num(row["life_expectancy"], 65.0), 0.1)
            a, b, c = st.columns(3)
            freedom = a.number_input("Freedom (0-1)", 0.0, 1.0, num(row["freedom"], 0.75), 0.01)
            generosity = b.number_input("Generosity", -0.5, 1.0, num(row["generosity"], 0.0), 0.01)
            corruption = c.number_input("Corruption (0-1)", 0.0, 1.0, num(row["corruption"], 0.75), 0.01)
            if st.form_submit_button("Save changes", type="primary"):
                db.update_yearly(u_country, int(u_year), {
                    "happiness": happiness, "gdp": gdp, "social_support": social,
                    "life_expectancy": life, "freedom": freedom,
                    "generosity": generosity, "corruption": corruption})
                refresh(f"Updated {u_country} {u_year}. See the trend line and radar in the {u_country} Focus tab.")

        st.markdown("**Change a country's 2024 ranking score** (its rank is recalculated)")
        rank_countries = sorted(raw_rank["country"])
        r_country = st.selectbox("Country", rank_countries, key="upd_rank_country",
                                 index=rank_countries.index(focus) if focus in rank_countries else 0)
        current = raw_rank.set_index("country")["happiness"]
        with st.form("update_ranking"):
            r_score = st.number_input("New 2024 happiness (0-10)", 0.0, 10.0,
                                      float(current[r_country]), 0.01,
                                      help=f"Current score: {current[r_country]:.2f}")
            if st.form_submit_button("Save score", type="primary"):
                db.update_ranking_score(r_country, r_score)
                refresh(f"{r_country}'s 2024 score is now {r_score:.2f}. "
                        f"Its new rank shows in the cards at the top and the Rankings tab.")

    # ---------- DELETE ----------
    elif action == "Delete":
        st.markdown("**Delete one yearly record**")
        a, b = st.columns(2)
        d_country = a.selectbox("Country", yearly_countries, key="del_country")
        d_year = b.selectbox("Year", sorted(raw[raw["country"] == d_country]["year"], reverse=True),
                             key="del_year")
        sure = st.checkbox(f"Yes, delete {d_country} {d_year}", key="del_one_ok")
        if st.button("Delete record", type="primary", disabled=not sure):
            db.delete_yearly(d_country, int(d_year))
            refresh(f"Deleted {d_country} {d_year}.")

        st.markdown("**Delete a whole country** (all yearly records and its 2024 ranking)")
        all_countries = sorted(set(yearly_countries) | set(raw_rank["country"]))
        dc = st.selectbox("Country to remove", all_countries, key="del_all_country")
        sure_all = st.checkbox(f"Yes, remove {dc} completely", key="del_all_ok")
        if st.button("Delete country", disabled=not sure_all):
            n = db.delete_country(dc)
            refresh(f"Removed {dc} ({n} rows).")

    # ---------- ACTIVITY LOG ----------
    elif action == "Activity log":
        st.dataframe(db.fetch_log(), hide_index=True)
        st.caption("Every insert, update and delete is recorded in the activity_log table.")

    # ---------- RESET ----------
    else:
        st.warning("This replaces everything in the database with the original Kaggle "
                   "data. All inserts, updates and deletes are undone (the activity log is kept).")
        sure_reset = st.checkbox("Yes, restore the original data", key="reset_ok")
        if st.button("Restore original data", type="primary", disabled=not sure_reset):
            if import_from_csv():
                refresh("Database restored from the Kaggle CSV files.")
            else:
                st.error("Could not find the Kaggle CSV files to restore from.")