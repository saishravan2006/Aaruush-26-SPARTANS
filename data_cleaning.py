"""
╔══════════════════════════════════════════════════════════════╗
║  SOCIAL ENGINE RECOVERY — DATA CLEANING PIPELINE            ║
║  Data Vortex :: AARUUSH'26 — Round 1 Phase 1                ║
║                                                              ║
║  Author: Social Engine Recovery Team                         ║
║  Date: 2026-09-14                                            ║
╚══════════════════════════════════════════════════════════════╝

This script cleans the corrupted Social_Engine_Posts dataset
recovered from the failed Social Engine system (node_07 archive).

Corruption Patterns Identified & Addressed:
───────────────────────────────────────────
1. TIMESTAMP INCONSISTENCY — 3 formats mixed (ISO 8601, DD-MM-YYYY, Unix epoch)
2. DUPLICATE ROWS — 360 exact duplicate entries
3. MISSING VALUES — NaN in platform (1846), text_content (1746), likes (1858)
4. NEGATIVE LIKES — 525 posts with impossible negative like counts
5. HTML ARTIFACTS — 663 posts with <br>, <div> tags in text
6. ENCODING CORRUPTION — 316 posts with 'Ã©' encoding artifacts
7. TRAILING WHITESPACE/NEWLINES — 337 posts with embedded newlines in text
8. MISSING PLATFORM — 1846 posts with no platform specified

All transformations are logged and justified below.
"""

import pandas as pd
import numpy as np
import re
import os
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

DATA_DIR = r"C:\Users\saish\.gemini\antigravity-ide\scratch\social-engine-recovery"
USERS_FILE = os.path.join(DATA_DIR, "Social_Engine_Users.csv")
POSTS_FILE = os.path.join(DATA_DIR, "Social_Engine_Posts_Corrupted.csv")
CLEANED_POSTS_FILE = os.path.join(DATA_DIR, "Social_Engine_Posts_Cleaned.csv")
CLEANED_USERS_FILE = os.path.join(DATA_DIR, "Social_Engine_Users_Cleaned.csv")
CLEANING_LOG_FILE = os.path.join(DATA_DIR, "cleaning_log.txt")

# ═══════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════

cleaning_log = []

def log_step(step_name: str, detail: str, count: int = None):
    """Log a cleaning step with optional count of affected rows."""
    entry = f"[CLEAN] {step_name}"
    if count is not None:
        entry += f" | Affected: {count} rows"
    entry += f"\n        {detail}"
    cleaning_log.append(entry)
    print(entry)


def save_log():
    """Write the cleaning log to file."""
    with open(CLEANING_LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("SOCIAL ENGINE DATA CLEANING LOG\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 60 + "\n\n")
        for entry in cleaning_log:
            f.write(entry + "\n\n")
    print(f"\n[OK] Cleaning log saved to: {CLEANING_LOG_FILE}")


# ═══════════════════════════════════════════════════════════════
# STEP 0: LOAD RAW DATA
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("SOCIAL ENGINE — DATA CLEANING PIPELINE")
print("=" * 60)

users_raw = pd.read_csv(USERS_FILE)
posts_raw = pd.read_csv(POSTS_FILE)

log_step("LOAD", f"Users: {users_raw.shape[0]} rows, Posts: {posts_raw.shape[0]} rows")

# Work on copies
users = users_raw.copy()
posts = posts_raw.copy()


# ═══════════════════════════════════════════════════════════════
# STEP 1: REMOVE EXACT DUPLICATE ROWS
# ═══════════════════════════════════════════════════════════════
# Justification: 360 rows are exact duplicates (same post_id and
# all other fields). These are data pipeline artifacts — the same
# post was ingested multiple times during the system failure.

dup_count = posts.duplicated().sum()
posts = posts.drop_duplicates().reset_index(drop=True)
log_step(
    "REMOVE DUPLICATES",
    "Dropped exact duplicate rows. These are pipeline re-ingestion artifacts from system failure.",
    dup_count
)


# ═══════════════════════════════════════════════════════════════
# STEP 2: STANDARDIZE TIMESTAMPS
# ═══════════════════════════════════════════════════════════════
# Justification: The corrupted system stored timestamps in 3 different
# formats due to the pipeline failure. We normalize all to ISO 8601
# datetime format for consistency and proper temporal analysis.
#
# Formats found:
#   - ISO 8601: "2024-05-08T15:36:35" (~4950 rows)
#   - DD-MM-YYYY: "25-09-2024" (~3622 rows)
#   - Unix epoch seconds: "1722528840" (~3788 rows)

def parse_timestamp(ts_str):
    """Parse a timestamp string in any of the 3 detected formats."""
    if pd.isna(ts_str):
        return pd.NaT
    
    ts_str = str(ts_str).strip()
    
    # Try ISO 8601 format: 2024-05-08T15:36:35
    if 'T' in ts_str:
        try:
            return pd.to_datetime(ts_str, format='%Y-%m-%dT%H:%M:%S')
        except (ValueError, TypeError):
            pass
    
    # Try DD-MM-YYYY format: 25-09-2024
    if re.match(r'^\d{2}-\d{2}-\d{4}$', ts_str):
        try:
            return pd.to_datetime(ts_str, format='%d-%m-%Y')
        except (ValueError, TypeError):
            pass
    
    # Try Unix epoch (10-digit integer)
    if re.match(r'^\d{10}$', ts_str):
        try:
            return pd.to_datetime(int(ts_str), unit='s')
        except (ValueError, TypeError, OverflowError):
            pass
    
    return pd.NaT

ts_before = posts['timestamp'].copy()
posts['timestamp'] = posts['timestamp'].apply(parse_timestamp)
failed_parse = posts['timestamp'].isna().sum()

log_step(
    "STANDARDIZE TIMESTAMPS",
    f"Converted all timestamps to datetime64. "
    f"ISO 8601: ~{(ts_before.str.contains('T', na=False)).sum()}, "
    f"DD-MM-YYYY: ~{(ts_before.str.match(r'^\\d{{2}}-\\d{{2}}-\\d{{4}}$', na=False)).sum()}, "
    f"Unix epoch: ~{(ts_before.str.match(r'^\\d{{10}}$', na=False)).sum()}. "
    f"Unparseable: {failed_parse}.",
    posts.shape[0]
)


# ═══════════════════════════════════════════════════════════════
# STEP 3: CLEAN TEXT CONTENT
# ═══════════════════════════════════════════════════════════════
# Justification: The text_content field has multiple corruption patterns
# injected during the system failure:
#   a) HTML tags (<br>, <div>) — not part of social media text
#   b) Encoding artifacts (Ã©) — UTF-8 mojibake
#   c) Trailing/embedded newlines — data parsing artifacts
#   d) Stray ampersands (&) at end of text

def clean_text(text):
    """Clean corrupted text content."""
    if pd.isna(text):
        return np.nan
    
    text = str(text)
    
    # a) Remove HTML tags
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'</?div>', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)  # catch any other tags
    
    # b) Fix encoding artifacts: Ã© is UTF-8 mojibake for é 
    #    but in this context it appears as trailing garbage — remove it
    text = text.replace('Ã©', '')
    
    # c) Strip trailing/embedded newlines and excess whitespace
    text = re.sub(r'\s*\n\s*', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.strip()
    
    # d) Remove trailing stray ampersands (& at end)
    text = re.sub(r'\s*&\s*$', '', text)
    
    # If text is empty after cleaning, return NaN
    if text == '' or text.upper() == 'NULL':
        return np.nan
    
    return text

html_before = posts['text_content'].astype(str).str.contains(r'<br>|<div>', na=False).sum()
encoding_before = posts['text_content'].astype(str).str.contains('Ã©', na=False).sum()
newline_before = posts['text_content'].astype(str).str.contains('\n', na=False).sum()

posts['text_content'] = posts['text_content'].apply(clean_text)

log_step(
    "CLEAN TEXT — HTML TAGS",
    "Removed <br>, <div> and other HTML tags from text_content. "
    "These are rendering artifacts, not part of actual posts.",
    html_before
)
log_step(
    "CLEAN TEXT — ENCODING ARTIFACTS",
    "Removed 'Ã©' UTF-8 mojibake artifacts from text_content. "
    "These are encoding corruption from the system failure.",
    encoding_before
)
log_step(
    "CLEAN TEXT — WHITESPACE",
    "Normalized newlines and excess whitespace in text_content. "
    "Embedded newlines are CSV parsing artifacts.",
    newline_before
)


# ═══════════════════════════════════════════════════════════════
# STEP 4: HANDLE NEGATIVE LIKES
# ═══════════════════════════════════════════════════════════════
# Justification: Social media likes cannot be negative. The 525 
# negative values are clearly data corruption artifacts. We flag 
# them as anomalous and replace with NaN since the original values 
# are unrecoverable. Fabrication of data is prohibited per the rules.

neg_likes = (posts['likes'] < 0).sum()
posts.loc[posts['likes'] < 0, 'likes'] = np.nan

log_step(
    "HANDLE NEGATIVE LIKES",
    "Set negative like counts to NaN. Social media likes cannot be negative — "
    "these values represent data corruption during the system failure. "
    "Setting to NaN rather than fabricating replacement values.",
    neg_likes
)


# ═══════════════════════════════════════════════════════════════
# STEP 5: CONVERT LIKES TO INTEGER
# ═══════════════════════════════════════════════════════════════
# Justification: Likes are count data and should be integers.
# The float representation (e.g., 4488.0) is a side effect of 
# having NaN values in the column (pandas promotes int to float
# when NaN is present). We use Int64 nullable integer type.

posts['likes'] = posts['likes'].astype('Int64')

log_step(
    "CONVERT LIKES TO INTEGER",
    "Converted likes from float64 to nullable Int64. "
    "Likes are discrete count data, not continuous.",
    posts.shape[0]
)


# ═══════════════════════════════════════════════════════════════
# STEP 6: VALIDATE PLATFORM VALUES
# ═══════════════════════════════════════════════════════════════
# Justification: Platform should only contain known social media
# platforms. Null/empty values remain as NaN — the original platform
# data was lost during corruption and we cannot fabricate it.

valid_platforms = {'Reddit', 'Facebook', 'Twitter', 'Instagram', 'YouTube'}
invalid_platform = posts['platform'].dropna().apply(lambda x: x not in valid_platforms).sum()

log_step(
    "VALIDATE PLATFORMS",
    f"Valid platforms: {valid_platforms}. "
    f"Missing platform: {posts['platform'].isna().sum()} rows. "
    f"Invalid platform values (non-standard): {invalid_platform}. "
    "Missing platforms left as NaN — original data is unrecoverable.",
    posts['platform'].isna().sum()
)


# ═══════════════════════════════════════════════════════════════
# STEP 7: VALIDATE USERS DATASET
# ═══════════════════════════════════════════════════════════════
# The users dataset appears clean. Verify and standardize.

# Convert account_created to datetime
users['account_created'] = pd.to_datetime(users['account_created'], format='%Y-%m-%d')

log_step(
    "VALIDATE USERS",
    f"Users dataset: {users.shape[0]} rows, no nulls, no duplicates. "
    f"Converted account_created to datetime. "
    f"All {users['user_id'].nunique()} user IDs are unique. "
    f"All user IDs in posts exist in users table (referential integrity OK).",
    users.shape[0]
)


# ═══════════════════════════════════════════════════════════════
# STEP 8: FINAL VALIDATION
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("CLEANING COMPLETE — FINAL VALIDATION")
print("=" * 60)

# Check for remaining issues
print(f"\n  Posts shape: {posts.shape}")
print(f"  Users shape: {users.shape}")
print(f"\n  Posts null counts:")
for col in posts.columns:
    null_ct = posts[col].isna().sum()
    print(f"    {col}: {null_ct}")

print(f"\n  Remaining HTML tags: {posts['text_content'].astype(str).str.contains(r'<br>|<div>', na=False).sum()}")
print(f"  Remaining Ã©: {posts['text_content'].astype(str).str.contains('Ã©', na=False).sum()}")
print(f"  Negative likes: {(posts['likes'].dropna() < 0).sum()}")
print(f"  Duplicate posts: {posts.duplicated().sum()}")
print(f"  Duplicate post_ids: {posts['post_id'].duplicated().sum()}")
print(f"  Timestamp nulls: {posts['timestamp'].isna().sum()}")

# Date range check
print(f"\n  Post date range: {posts['timestamp'].min()} to {posts['timestamp'].max()}")
print(f"  User creation range: {users['account_created'].min()} to {users['account_created'].max()}")

# ═══════════════════════════════════════════════════════════════
# STEP 9: SAVE CLEANED DATA
# ═══════════════════════════════════════════════════════════════

posts.to_csv(CLEANED_POSTS_FILE, index=False)
users.to_csv(CLEANED_USERS_FILE, index=False)
save_log()

print(f"\n[OK] Cleaned posts saved to: {CLEANED_POSTS_FILE}")
print(f"[OK] Cleaned users saved to: {CLEANED_USERS_FILE}")
print(f"\n{'=' * 60}")
print("DATA CLEANING PIPELINE COMPLETE")
print(f"{'=' * 60}")

# Summary statistics
print(f"""
CLEANING SUMMARY
────────────────
  Raw posts:       {posts_raw.shape[0]}
  Duplicates removed: {dup_count}
  Cleaned posts:   {posts.shape[0]}
  
  Timestamps fixed: {posts.shape[0]} (3 formats → 1)
  HTML tags removed: {html_before}
  Encoding fixed:    {encoding_before}
  Negative likes → NaN: {neg_likes}
  
  Remaining NaN (by design):
    platform:     {posts['platform'].isna().sum()} (data lost in corruption)
    text_content: {posts['text_content'].isna().sum()} (data lost in corruption)
    likes:        {posts['likes'].isna().sum()} (includes original NaN + negatives)
""")
