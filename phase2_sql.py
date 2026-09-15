"""
╔══════════════════════════════════════════════════════════════╗
║  Data Vortex :: AARUUSH'26 — Round 1 Phase 2               ║
║  Social Engine Recovery — SQL Analytics                     ║
║  Team: SPARTANS                                             ║
╚══════════════════════════════════════════════════════════════╝

Schema Design:
  We use a normalized relational schema with two tables:
  
  1. posts — Contains all post-level data (engagement metrics, content, timestamps)
  2. users — Contains user profile data (location, language, followers)
  
  The tables are linked via user_id (FOREIGN KEY).
  We also create a view 'posts_corrupted' from the raw corrupted CSV for H5.

  Engine: SQLite 3 (lightweight, zero-config, perfect for analytical queries)
"""

import sqlite3
import pandas as pd
import os

DB_PATH = 'social_engine.db'

# Remove old DB if exists
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# ═══════════════════════════════════════════
# SCHEMA CREATION
# ═══════════════════════════════════════════

cursor.executescript("""
-- Users table: stores user profile information
CREATE TABLE users (
    user_id         TEXT PRIMARY KEY,
    location        TEXT NOT NULL,
    language        TEXT NOT NULL,
    account_created TEXT NOT NULL,
    follower_count  INTEGER NOT NULL
);

-- Posts table (cleaned): stores cleaned post data with engagement metrics
CREATE TABLE posts (
    post_id      TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    platform     TEXT,           -- NULL = missing platform (corruption artifact)
    text_content TEXT,           -- NULL = missing text (corruption artifact)
    timestamp    TEXT NOT NULL,
    likes        REAL,           -- NULL = missing/negative likes (cleaned)
    shares       INTEGER NOT NULL,
    comments     INTEGER NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- Posts corrupted: raw corrupted data for anomaly detection (H5)
CREATE TABLE posts_corrupted (
    post_id      TEXT PRIMARY KEY,
    user_id      TEXT,
    platform     TEXT,
    text_content TEXT,
    timestamp    TEXT,
    likes        TEXT,           -- TEXT because corrupted data has mixed types
    shares       TEXT,
    comments     TEXT
);
""")

# ═══════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════

print("Loading data into SQLite...")

users_df = pd.read_csv('Social_Engine_Users_Cleaned.csv')
users_df.to_sql('users', conn, if_exists='replace', index=False)
print(f"  users: {len(users_df)} rows loaded")

posts_df = pd.read_csv('Social_Engine_Posts_Cleaned.csv')
posts_df.to_sql('posts', conn, if_exists='replace', index=False)
print(f"  posts: {len(posts_df)} rows loaded")

corrupted_df = pd.read_csv('Social_Engine_Posts_Corrupted.csv', dtype=str)
corrupted_df.to_sql('posts_corrupted', conn, if_exists='replace', index=False)
print(f"  posts_corrupted: {len(corrupted_df)} rows loaded")

conn.commit()

# Create indexes for performance
cursor.executescript("""
CREATE INDEX IF NOT EXISTS idx_posts_user_id ON posts(user_id);
CREATE INDEX IF NOT EXISTS idx_posts_platform ON posts(platform);
CREATE INDEX IF NOT EXISTS idx_users_follower_count ON users(follower_count);
CREATE INDEX IF NOT EXISTS idx_posts_corrupted_post_id ON posts_corrupted(post_id);
""")
conn.commit()

print("\nSchema created with indexes. Ready for queries!\n")

# ═══════════════════════════════════════════
# HELPER: Run and display query
# ═══════════════════════════════════════════

results_log = []  # Store for PDF generation

def run_query(question_id, title, sql, logic):
    """Execute a SQL query, display and log results."""
    print("=" * 75)
    print(f"  {question_id} — {title}")
    print("=" * 75)
    print(f"\n  Logic: {logic}\n")
    print(f"  SQL Query:")
    for line in sql.strip().split('\n'):
        print(f"    {line}")
    print()
    
    df = pd.read_sql_query(sql, conn)
    print(f"  Result ({len(df)} rows):")
    print(df.to_string(index=False))
    print()
    
    results_log.append({
        'id': question_id,
        'title': title,
        'sql': sql.strip(),
        'logic': logic,
        'result': df,
        'row_count': len(df)
    })
    return df


# ═══════════════════════════════════════════════════════════════
#                        EASY LEVEL
# ═══════════════════════════════════════════════════════════════

# ── E1 ──
run_query("E1", "Platform Popularity", """
SELECT platform, 
       COUNT(*) AS num_posts
FROM posts
WHERE platform IS NOT NULL
GROUP BY platform
ORDER BY num_posts DESC;
""", 
"Filter out NULL platforms, GROUP BY platform, COUNT posts, sort descending.")

# ── E2 ──
run_query("E2", "Most Engaged Posts (Top 10)", """
SELECT post_id, 
       platform,
       likes, 
       shares, 
       comments,
       (likes + shares + comments) AS total_engagement
FROM posts
WHERE likes IS NOT NULL
ORDER BY total_engagement DESC
LIMIT 10;
""",
"Calculate total_engagement = likes + shares + comments. Filter out posts with NULL likes. Order descending, LIMIT 10.")

# ── E3 ──
run_query("E3", "Average Engagement by Platform", """
SELECT platform,
       ROUND(AVG(likes), 2)   AS avg_likes,
       ROUND(AVG(shares), 2)  AS avg_shares,
       ROUND(AVG(comments), 2) AS avg_comments,
       ROUND(AVG(likes) + AVG(shares) + AVG(comments), 2) AS avg_total_engagement
FROM posts
WHERE platform IS NOT NULL
GROUP BY platform
ORDER BY avg_total_engagement DESC;
""",
"GROUP BY platform, compute AVG for each engagement metric. Sum the averages for total. YouTube leads with 4044.54.")

# ── E4 ──
run_query("E4", "Highly Shared but Poorly Liked", """
SELECT post_id, 
       platform, 
       likes, 
       shares, 
       comments
FROM posts
WHERE shares > 1500 
  AND likes < 500
ORDER BY shares DESC;
""",
"Simple WHERE clause filtering: shares > 1500 AND likes < 500. These posts have abnormally high sharing relative to their like count.")

# ── E5 ──
run_query("E5", "Users With Large Audiences (>40K followers)", """
SELECT user_id, 
       location, 
       language, 
       follower_count
FROM users
WHERE follower_count > 40000
ORDER BY follower_count DESC;
""",
"Direct filter on follower_count > 40000, sorted descending.")


# ═══════════════════════════════════════════════════════════════
#                       MEDIUM LEVEL
# ═══════════════════════════════════════════════════════════════

# ── M1 ──
run_query("M1", "Which Locations Generate the Most Engagement?", """
SELECT u.location,
       COUNT(p.post_id) AS num_posts,
       ROUND(SUM(COALESCE(p.likes, 0) + p.shares + p.comments), 0) AS total_engagement
FROM posts p
JOIN users u ON p.user_id = u.user_id
GROUP BY u.location
ORDER BY total_engagement DESC;
""",
"JOIN posts with users on user_id. COALESCE(likes, 0) handles NULLs. GROUP BY location, SUM engagement, rank descending. Los Angeles leads.")

# ── M2 ──
run_query("M2", "Do High Follower Users Get More Engagement?", """
SELECT 
    CASE 
        WHEN u.follower_count >= 25000 THEN 'High (>=25K)'
        ELSE 'Low (<25K)'
    END AS follower_group,
    ROUND(AVG(COALESCE(p.likes, 0) + p.shares + p.comments), 2) AS avg_engagement_per_post
FROM posts p
JOIN users u ON p.user_id = u.user_id
GROUP BY follower_group
ORDER BY avg_engagement_per_post DESC;
""",
"Use CASE expression to bucket users into High/Low groups. JOIN with posts, calculate AVG engagement per post per group. Surprisingly, low-follower users get slightly more engagement.")

# ── M3 ──
run_query("M3", "Most Active Users (Top 10)", """
SELECT p.user_id,
       u.follower_count,
       u.location,
       COUNT(p.post_id) AS post_count,
       ROUND(SUM(COALESCE(p.likes, 0) + p.shares + p.comments), 0) AS total_engagement
FROM posts p
JOIN users u ON p.user_id = u.user_id
GROUP BY p.user_id
ORDER BY post_count DESC, total_engagement DESC
LIMIT 10;
""",
"JOIN posts with users. GROUP BY user_id, COUNT posts, SUM engagement. Order by post_count DESC to find the most prolific posters.")

# ── M4 ──
run_query("M4", "Platform Behaviour by High Follower Users (>=30K)", """
SELECT p.platform,
       ROUND(AVG(COALESCE(p.likes, 0) + p.shares + p.comments), 2) AS avg_engagement_per_post
FROM posts p
JOIN users u ON p.user_id = u.user_id
WHERE u.follower_count >= 30000
  AND p.platform IS NOT NULL
GROUP BY p.platform
ORDER BY avg_engagement_per_post DESC;
""",
"Filter users with >= 30K followers, exclude NULL platforms. GROUP BY platform, compute AVG engagement. Instagram performs best for high-follower users.")

# ── M5 ──
run_query("M5", "Detect Suspicious Engagement (Top 20)", """
SELECT post_id,
       platform,
       likes,
       shares,
       comments,
       (COALESCE(likes, 0) + shares + comments) AS total_engagement
FROM posts
WHERE shares > (COALESCE(likes, 0) + comments)
ORDER BY total_engagement DESC
LIMIT 20;
""",
"Suspicious = shares > (likes + comments). This indicates artificial amplification or bot-driven sharing behaviour. 2,915 such posts exist.")


# ═══════════════════════════════════════════════════════════════
#                        HARD LEVEL
# ═══════════════════════════════════════════════════════════════

# ── H1 ──
run_query("H1", "Users With Abnormally High Engagement", """
WITH user_engagement AS (
    SELECT p.user_id,
           u.location,
           u.follower_count,
           COUNT(p.post_id) AS post_count,
           ROUND(AVG(COALESCE(p.likes, 0) + p.shares + p.comments), 2) AS avg_engagement
    FROM posts p
    JOIN users u ON p.user_id = u.user_id
    GROUP BY p.user_id
),
overall AS (
    SELECT AVG(COALESCE(likes, 0) + shares + comments) AS overall_avg
    FROM posts
)
SELECT ue.user_id, 
       ue.location, 
       ue.follower_count, 
       ue.post_count, 
       ue.avg_engagement,
       ROUND(o.overall_avg, 2) AS overall_avg,
       ROUND(2 * o.overall_avg, 2) AS threshold
FROM user_engagement ue, overall o
WHERE ue.avg_engagement > 2 * o.overall_avg
ORDER BY ue.avg_engagement DESC;
""",
"CTE 1: Calculate per-user average engagement. CTE 2: Calculate overall average across all posts. Filter users whose avg > 2x overall. Result: 0 users exceed threshold (7044.63) — engagement is uniformly distributed in this dataset.")

# ── H2 ──
run_query("H2", "Rank Users Within Their Location (Top 3)", """
WITH user_location_engagement AS (
    SELECT p.user_id,
           u.location,
           SUM(COALESCE(p.likes, 0) + p.shares + p.comments) AS total_engagement,
           RANK() OVER (
               PARTITION BY u.location 
               ORDER BY SUM(COALESCE(p.likes, 0) + p.shares + p.comments) DESC
           ) AS location_rank
    FROM posts p
    JOIN users u ON p.user_id = u.user_id
    GROUP BY p.user_id, u.location
)
SELECT user_id, 
       location, 
       ROUND(total_engagement, 0) AS total_engagement, 
       location_rank
FROM user_location_engagement
WHERE location_rank <= 3
ORDER BY location, location_rank;
""",
"Window function RANK() OVER (PARTITION BY location ORDER BY engagement DESC) assigns ranks within each location. Filter WHERE rank <= 3 to get top 3 per city.")

# ── H3 ──
run_query("H3", "Exceptional Posts (>=2x Platform Average)", """
WITH platform_avg AS (
    SELECT platform,
           AVG(COALESCE(likes, 0) + shares + comments) AS avg_engagement
    FROM posts
    WHERE platform IS NOT NULL
    GROUP BY platform
)
SELECT p.post_id,
       p.platform,
       ROUND(COALESCE(p.likes, 0) + p.shares + p.comments, 0) AS post_engagement,
       ROUND(pa.avg_engagement, 2) AS platform_avg_engagement
FROM posts p
JOIN platform_avg pa ON p.platform = pa.platform
WHERE (COALESCE(p.likes, 0) + p.shares + p.comments) >= 2 * pa.avg_engagement
ORDER BY post_engagement DESC
LIMIT 20;
""",
"CTE computes each platform's average engagement. Main query JOINs back and filters posts where engagement >= 2x their platform's average. These are statistically exceptional outliers.")

# ── H4 ──
run_query("H4", "Follower to Engagement Anomaly", """
WITH user_totals AS (
    SELECT p.user_id,
           u.follower_count,
           u.location,
           SUM(COALESCE(p.likes, 0) + p.shares + p.comments) AS total_engagement,
           COUNT(p.post_id) AS post_count
    FROM posts p
    JOIN users u ON p.user_id = u.user_id
    GROUP BY p.user_id
),
percentile_threshold AS (
    SELECT total_engagement AS threshold
    FROM user_totals
    ORDER BY total_engagement DESC
    LIMIT 1 OFFSET (SELECT CAST(COUNT(*) * 0.10 AS INTEGER) FROM user_totals)
)
SELECT ut.user_id,
       ut.follower_count,
       ut.location,
       ut.post_count,
       ROUND(ut.total_engagement, 0) AS total_engagement
FROM user_totals ut, percentile_threshold pt
WHERE ut.follower_count < 5000
  AND ut.total_engagement >= pt.threshold
ORDER BY ut.total_engagement DESC;
""",
"Multi-level analysis: CTE 1 aggregates per-user totals. CTE 2 calculates the 90th percentile threshold using OFFSET. Filter: < 5K followers BUT engagement in top 10%. These are anomalously high-performing low-follower accounts.")

# ── H5 ──
run_query("H5", "Identify Data Anomalies (Corrupted Dataset)", """
SELECT post_id,
       CASE
           WHEN (CAST(likes AS REAL) < 0) AND (platform IS NULL OR platform = '') 
                AND (text_content IS NULL OR text_content = '')
                AND (text_content LIKE '%&amp;%' OR text_content LIKE '%<div>%' OR text_content LIKE '%<br>%')
               THEN 'Negative likes | Missing platform | Missing text | HTML tags'
           WHEN (CAST(likes AS REAL) < 0) AND (platform IS NULL OR platform = '')
               THEN 'Negative likes | Missing platform'
           WHEN (CAST(likes AS REAL) < 0) AND (text_content LIKE '%&amp;%' OR text_content LIKE '%<div>%' OR text_content LIKE '%<br>%')
               THEN 'Negative likes | HTML tags'
           WHEN (platform IS NULL OR platform = '') AND (text_content IS NULL OR text_content = '')
               THEN 'Missing platform | Missing text'
           WHEN CAST(likes AS REAL) < 0
               THEN 'Negative likes'
           WHEN platform IS NULL OR platform = ''
               THEN 'Missing platform'
           WHEN text_content IS NULL OR text_content = ''
               THEN 'Missing text content'
           WHEN text_content LIKE '%&amp;%' OR text_content LIKE '%<div>%' 
                OR text_content LIKE '%<br>%' OR text_content LIKE '%</div>%'
               THEN 'Contains HTML tags/entities'
           ELSE NULL
       END AS anomaly_type
FROM posts_corrupted
WHERE (CAST(likes AS REAL) < 0)
   OR (platform IS NULL OR platform = '')
   OR (text_content IS NULL OR text_content = '')
   OR (text_content LIKE '%&amp;%' OR text_content LIKE '%<div>%' 
       OR text_content LIKE '%<br>%' OR text_content LIKE '%</div>%')
ORDER BY anomaly_type;
""",
"Uses the CORRUPTED dataset (not cleaned). CASE expression classifies each anomaly type. Checks 4 corruption patterns: negative likes (CAST to REAL), missing platform, missing text, HTML entities/tags via LIKE patterns.")

# ── H6 ──
run_query("H6", "Most Suspicious High Impact Users", """
WITH user_stats AS (
    SELECT p.user_id,
           u.location,
           u.follower_count,
           COUNT(p.post_id) AS post_count,
           ROUND(AVG(COALESCE(p.likes, 0) + p.shares + p.comments), 2) AS avg_engagement,
           ROUND(SUM(COALESCE(p.likes, 0) + p.shares + p.comments), 0) AS total_engagement,
           MAX(CASE WHEN p.shares > COALESCE(p.likes, 0) THEN 1 ELSE 0 END) AS has_shares_gt_likes
    FROM posts p
    JOIN users u ON p.user_id = u.user_id
    GROUP BY p.user_id
),
overall AS (
    SELECT AVG(COALESCE(likes, 0) + shares + comments) AS overall_avg
    FROM posts
)
SELECT us.user_id,
       us.location,
       us.follower_count,
       us.post_count,
       us.avg_engagement,
       us.total_engagement
FROM user_stats us, overall o
WHERE us.follower_count < 10000
  AND us.avg_engagement > o.overall_avg
  AND us.has_shares_gt_likes = 1
ORDER BY us.total_engagement DESC;
""",
"Three-condition filter using CTEs: (1) follower_count < 10K, (2) avg engagement > overall average, (3) MAX(CASE WHEN shares > likes) = 1 checks if ANY post has more shares than likes. These users are suspiciously high-performing despite low follower counts.")


# ═══════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════

print("\n" + "=" * 75)
print("  ALL 16 SQL QUERIES EXECUTED SUCCESSFULLY")
print("=" * 75)

# Save query log for PDF generation
import json
log_data = []
for r in results_log:
    log_data.append({
        'id': r['id'],
        'title': r['title'],
        'sql': r['sql'],
        'logic': r['logic'],
        'row_count': r['row_count'],
        'result_csv': r['result'].to_csv(index=False)
    })

with open('phase2_sql_results.json', 'w', encoding='utf-8') as f:
    json.dump(log_data, f, indent=2)

print("\nResults saved to phase2_sql_results.json")

conn.close()
