"""
Merge BPS school datasets into a single clean CSV for RAG use.

Sources:
  - dese_bps_all_schools.csv  (base: enrollment, grades, demographics, contact)
  - raw data/public_schools.csv  (adds: lat/long, zipcode, school type code)
  - raw data/bps_special_education_programs.csv  (adds: sped program list per school)

Output: data_cleaning/bps_clean.csv  (one row per school)

Run from repo root:
  /Users/willreed/miniconda3/bin/python3 data_cleaning/data_merge.py
"""

import os
import pandas as pd
from rapidfuzz import process, fuzz

DATA_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.join(DATA_DIR, "raw data")
OUTPUT = os.path.join(DATA_DIR, "bps_clean.csv")


# ── helpers ──────────────────────────────────────────────────────────────────

def normalize(name: str) -> str:
    """Lowercase, strip punctuation/common words for fuzzy matching."""
    s = str(name).lower().strip()
    # Expand abbreviations before stripping
    s = s.replace("eec", "early education center").replace("elc", "early learning center")
    # Strip grade-range suffixes that differ between datasets
    s = s.replace("k-8", "").replace("k-12", "").replace("k8", "").replace("k12", "")
    # Strip descriptor words that are inconsistently used
    for word in ("school", "elementary", "pilot", "innovation", "inclusion",
                 "technical", "vocational", "stem", "newcomers", "the "):
        s = s.replace(word, "")
    # Normalize punctuation
    s = s.replace(".", "").replace(",", "").replace("-", " ").replace("/", " ").strip()
    return s


# Manual overrides for schools with genuinely different names between datasets.
# Key = DESE school_name, Value = exact SCH_NAME in public_schools.csv
GEO_OVERRIDES: dict[str, str] = {
    "Albert D Holland School of Technology":          "Holland Elementary",
    "Haynes Early Education Center":                  "Haynes EEC",
    "East Boston Early Education Center":             "East Boston EEC",
    "Baldwin Early Learning Pilot Academy":           "Baldwin ELPA",
    "Warren-Prescott K-8 School":                     "Warren/Prescott",
    "Madison Park Technical Vocational High School":  "Madison Park High",
    "Community Academy of Science and Health":        "Comm Acad Sci Health",
    "Ellison-Parks Early Education School":           "Ellison/ Parks ELC",
    "Henderson K-12 Inclusion School Lower":          "Henderson Lower (K-3)",
    "Henderson K-12 Inclusion School Upper":          "Henderson Upper (4-12)",
    "Dearborn STEM Academy":                          "Dearborn Middle School",
    "Mario Umana Academy":                            "Umana/ Alighieri K-8",
    "Gardner Pilot Academy":                          "Gardner Elementary",
    "Boston Teachers Union Elementary Pilot School":  "BTU K-8 Pilot",
    "Higginson Inclusion K0-2 School":                "Higginson Elementary",
    "Boston International High School & Newcomers Academy": "Boston International HS",
    "Horace Mann School for the Deaf Hard of Hearing": "Horace Mann",
    "Lyon High School":                               "Lyon, Mary 9-12",
    "Kilmer K-8 School":                              "Kilmer Upper (4-8)",
    "Melvin H. King South End Academy":               "McKinley So. End Acad",
    "Carter School":                                  "Carter Center",
    "Curley K-8 School":                              "Curley Upper (6-8)",
    "UP Academy Holland":                             "UP Academy",
    "Quincy Elementary School":                       "Quincy Lower (K-5)",
    "Roosevelt K-8 School":                           "Roosevelt K-8 (K1-1)",
}


def best_match(name: str, choices: list[str], threshold: int = 75) -> str | None:
    result = process.extractOne(name, choices, scorer=fuzz.token_sort_ratio)
    if result and result[1] >= threshold:
        return result[0]
    return None


# ── load ─────────────────────────────────────────────────────────────────────

dese = pd.read_csv(os.path.join(DATA_DIR, "dese_bps_all_schools.csv"))
geo = pd.read_csv(os.path.join(RAW_DIR, "public_schools.csv"), encoding="utf-8-sig")
sped = pd.read_csv(os.path.join(RAW_DIR, "bps_special_education_programs.csv"))

print(f"Loaded: {len(dese)} DESE rows, {len(geo)} geo rows, {len(sped)} sped rows")

# ── clean DESE (base table) ───────────────────────────────────────────────────

dese = dese.rename(columns={
    "org_code": "dese_org_code",
    "school_name": "school_name",
    "school_address": "address",
    "school_phone_number": "phone",
    "principal_name": "principal_name",
    "principal_email": "principal_email",
    "school_type": "school_type",
    "enrollment": "enrollment",
    "grades_served": "grades_served",
    "student_teacher_ratio": "student_teacher_ratio",
    "pop_english_learners_pct": "pct_english_learners",
    "pop_students_with_disabilities_pct": "pct_disabilities",
    "pop_low_income_pct": "pct_low_income",
    "pop_high_needs_pct": "pct_high_needs",
})

# Keep only columns useful for the chatbot
dese_cols = [
    "dese_org_code", "school_name", "address", "phone",
    "principal_name", "principal_email",
    "grades_served", "enrollment", "student_teacher_ratio",
    "pct_english_learners", "pct_disabilities", "pct_low_income", "pct_high_needs",
]
dese = dese[dese_cols].copy()
dese["name_key"] = dese["school_name"].apply(normalize)

# ── merge geo (lat/long, zipcode, school type code) ───────────────────────────

geo = geo.rename(columns={
    "SCH_NAME": "geo_name",
    "SCH_TYPE": "school_type_code",  # ES / MS / HS / K8 / etc.
    "ZIPCODE": "zipcode",
    "POINT_X": "longitude",
    "POINT_Y": "latitude",
    "ADDRESS": "street_address",
    "CITY": "city",
})
geo = geo[["geo_name", "school_type_code", "zipcode", "latitude", "longitude"]].copy()
geo = geo.drop_duplicates(subset=["geo_name"])  # geo file has some duplicate entries
geo["name_key"] = geo["geo_name"].apply(normalize)

geo_name_keys = geo["name_key"].tolist()
geo_name_lookup = dict(zip(geo["geo_name"], geo["name_key"]))  # geo_name → name_key

def match_geo(row) -> str | None:
    # 1. Check manual overrides first (exact DESE name → geo name)
    geo_name = GEO_OVERRIDES.get(row["school_name"])
    if geo_name and geo_name in geo_name_lookup:
        return geo_name_lookup[geo_name]
    # 2. Fall back to fuzzy match on normalized keys
    return best_match(row["name_key"], geo_name_keys, threshold=75)

dese["geo_match_key"] = dese.apply(match_geo, axis=1)
dese = dese.merge(
    geo[["name_key", "school_type_code", "zipcode", "latitude", "longitude"]],
    left_on="geo_match_key",
    right_on="name_key",
    how="left",
    suffixes=("", "_geo"),
).drop(columns=["name_key_geo", "geo_match_key"])

geo_matched = dese["school_type_code"].notna().sum()
print(f"Geo matched: {geo_matched}/{len(dese)} schools")

# ── merge sped programs (many-to-one → list per school) ──────────────────────

# Filter out non-program rows (Public Day Schools / Special Admission are categories, not programs)
EXCLUDE_TYPES = {"Public Day Schools", "Special Admission Schools"}
sped_programs = sped[~sped["Program_Type"].isin(EXCLUDE_TYPES)].copy()
sped_programs["name_key"] = sped_programs["School Name"].apply(normalize)

dese_name_keys = dese["name_key"].tolist()

def match_dese(name_key):
    return best_match(name_key, dese_name_keys, threshold=75)

sped_programs["dese_match_key"] = sped_programs["name_key"].apply(match_dese)
sped_programs = sped_programs.dropna(subset=["dese_match_key"])

# Group: one row per school with a list of sped program types
sped_grouped = (
    sped_programs.groupby("dese_match_key")["Program_Type"]
    .apply(lambda x: "; ".join(sorted(set(x))))
    .reset_index()
    .rename(columns={"Program_Type": "special_education_programs", "dese_match_key": "name_key"})
)

dese = dese.merge(sped_grouped, on="name_key", how="left")

sped_matched = dese["special_education_programs"].notna().sum()
print(f"Sped programs matched: {sped_matched}/{len(dese)} schools")

# ── finalize ──────────────────────────────────────────────────────────────────

dese = dese.drop(columns=["name_key"])

# Reorder columns for readability
final_cols = [
    "dese_org_code",
    "school_name",
    "address",
    "zipcode",
    "latitude",
    "longitude",
    "phone",
    "principal_name",
    "principal_email",
    "school_type_code",
    "grades_served",
    "enrollment",
    "student_teacher_ratio",
    "pct_english_learners",
    "pct_disabilities",
    "pct_low_income",
    "pct_high_needs",
    "special_education_programs",
]
dese = dese[final_cols]

# Preserve leading zeros in zipcode (02128, not 2128.0)
dese["zipcode"] = dese["zipcode"].apply(
    lambda x: f"{int(x):05d}" if pd.notna(x) else ""
)

dese.to_csv(OUTPUT, index=False)
print(f"\nWrote {len(dese)} rows → {OUTPUT}")

# ── QA report ────────────────────────────────────────────────────────────────

print("\n── Missing value report ──")
for col in final_cols:
    missing = dese[col].isna().sum()
    if missing:
        print(f"  {col}: {missing} missing")

print("\n── Schools with NO geo match (no lat/long or type code) ──")
no_geo = dese[dese["latitude"].isna()][["school_name", "grades_served"]].head(15)
print(no_geo.to_string(index=False))

print("\n── Schools with sped programs ──")
print(dese[dese["special_education_programs"].notna()][["school_name", "special_education_programs"]].head(10).to_string(index=False))
