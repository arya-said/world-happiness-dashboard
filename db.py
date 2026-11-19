"""
DATABASE LAYER (SQLite) for the World Happiness Dashboard
SQLite is a small database that lives in ONE file (happiness.db).
It is built into Python (module: sqlite3), so nothing to install.

Tables:
  ranking       -> 2024 ranking (one row per country)
  yearly        -> yearly data 2005-2023 (one row per country + year)
  activity_log  -> every insert / update / delete done from the app

All queries use "?" placeholders (parameterised queries), which
protects against SQL injection.
"""

import os
import sqlite3
from datetime import datetime

import pandas as pd

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "happiness.db")

# Columns stored in the yearly table (database name -> CSV name)
YEARLY_COLS = {
    "country": "Country name",
    "year": "year",
    "happiness": "Life Ladder",
    "gdp": "Log GDP per capita",
    "social_support": "Social support",
    "life_expectancy": "Healthy life expectancy at birth",
    "freedom": "Freedom to make life choices",
    "generosity": "Generosity",
    "corruption": "Perceptions of corruption",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS ranking (
    country    TEXT PRIMARY KEY,
    region     TEXT,
    happiness  REAL NOT NULL CHECK (happiness BETWEEN 0 AND 10)
);
CREATE TABLE IF NOT EXISTS yearly (
    country          TEXT    NOT NULL,
    year             INTEGER NOT NULL,
    happiness        REAL    NOT NULL CHECK (happiness BETWEEN 0 AND 10),
    gdp              REAL,
    social_support   REAL,
    life_expectancy  REAL,
    freedom          REAL,
    generosity       REAL,
    corruption       REAL,
    PRIMARY KEY (country, year)
);
CREATE TABLE IF NOT EXISTS activity_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    action     TEXT NOT NULL,
    details    TEXT NOT NULL,
    done_at    TEXT NOT NULL
);
"""


# ------------------------------------------------------------------
# Connection + setup
# ------------------------------------------------------------------
def get_conn():
    """Open a connection to the database file (created if missing)."""
    return sqlite3.connect(DB_PATH)


def is_empty():
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        return conn.execute("SELECT COUNT(*) FROM yearly").fetchone()[0] == 0


def import_csvs(ranking_csv_df, yearly_csv_df):
    """Fill the database from the two Kaggle CSV DataFrames (first run / reset)."""
    rank = pd.DataFrame({
        "country": ranking_csv_df["Country name"],
        "region": ranking_csv_df["Regional indicator"],
        "happiness": ranking_csv_df["Ladder score"],
    }).dropna(subset=["happiness"])
    year = yearly_csv_df[list(YEARLY_COLS.values())].copy()
    year.columns = list(YEARLY_COLS.keys())

    with get_conn() as conn:
        conn.executescript(SCHEMA)
        conn.execute("DELETE FROM ranking")
        conn.execute("DELETE FROM yearly")
        rank.to_sql("ranking", conn, if_exists="append", index=False)
        year.to_sql("yearly", conn, if_exists="append", index=False)
        log(conn, "IMPORT", f"Loaded {len(rank)} ranking rows and {len(year)} yearly rows from CSV")


def log(conn, action, details):
    conn.execute("INSERT INTO activity_log (action, details, done_at) VALUES (?, ?, ?)",
                 (action, details, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


# ------------------------------------------------------------------
# READ
# ------------------------------------------------------------------
def fetch_ranking():
    with get_conn() as conn:
        return pd.read_sql("SELECT country, region, happiness FROM ranking", conn)


def fetch_yearly():
    with get_conn() as conn:
        return pd.read_sql("SELECT * FROM yearly ORDER BY country, year", conn)


def fetch_log(limit=50):
    with get_conn() as conn:
        return pd.read_sql("SELECT id, action, details, done_at FROM activity_log "
                           "ORDER BY id DESC LIMIT ?", conn, params=(limit,))


# ------------------------------------------------------------------
# INSERT
# ------------------------------------------------------------------
def insert_yearly(record):
    """Add one new country-year row. Raises sqlite3.IntegrityError if it already exists."""
    cols = list(YEARLY_COLS.keys())
    sql = f"INSERT INTO yearly ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})"
    with get_conn() as conn:
        conn.execute(sql, [record[c] for c in cols])
        log(conn, "INSERT", f"{record['country']} {record['year']}: happiness {record['happiness']}")


def insert_ranking(country, region, happiness):
    """Add a new country to the 2024 ranking. Raises IntegrityError if it is already there."""
    with get_conn() as conn:
        conn.execute("INSERT INTO ranking (country, region, happiness) VALUES (?, ?, ?)",
                     (country, region, happiness))
        log(conn, "INSERT", f"{country} added to 2024 ranking ({region}): happiness {happiness}")


# ------------------------------------------------------------------
# UPDATE
# ------------------------------------------------------------------
def update_yearly(country, year, values):
    """Change the values of an existing country-year row."""
    sets = ", ".join(f"{c} = ?" for c in values)
    with get_conn() as conn:
        cur = conn.execute(f"UPDATE yearly SET {sets} WHERE country = ? AND year = ?",
                           [*values.values(), country, year])
        log(conn, "UPDATE", f"{country} {year}: " +
            ", ".join(f"{k} = {v}" for k, v in values.items()))
        return cur.rowcount


def update_ranking_score(country, happiness):
    """Change a country's 2024 happiness score (its rank is recalculated by the app)."""
    with get_conn() as conn:
        cur = conn.execute("UPDATE ranking SET happiness = ? WHERE country = ?", (happiness, country))
        log(conn, "UPDATE", f"{country} 2024 ranking score -> {happiness}")
        return cur.rowcount


# ------------------------------------------------------------------
# DELETE
# ------------------------------------------------------------------
def delete_yearly(country, year):
    """Remove one country-year row."""
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM yearly WHERE country = ? AND year = ?", (country, year))
        log(conn, "DELETE", f"{country} {year}")
        return cur.rowcount


def delete_country(country):
    """Remove a country completely (all its yearly rows and its 2024 ranking row)."""
    with get_conn() as conn:
        n1 = conn.execute("DELETE FROM yearly WHERE country = ?", (country,)).rowcount
        n2 = conn.execute("DELETE FROM ranking WHERE country = ?", (country,)).rowcount
        log(conn, "DELETE", f"{country}: {n1} yearly rows and {n2} ranking row removed")
        return n1 + n2