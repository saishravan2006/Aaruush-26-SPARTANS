"""
╔══════════════════════════════════════════════════════════════╗
║  Data Vortex :: AARUUSH'26 — Round 1 Phase 2               ║
║  Social Engine Recovery — Questionnaire Solutions           ║
║  Team: SPARTANS                                             ║
╚══════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Load Data ──
posts_clean = pd.read_csv('Social_Engine_Posts_Cleaned.csv')
users = pd.read_csv('Social_Engine_Users_Cleaned.csv')
posts_corrupted = pd.read_csv('Social_Engine_Posts_Corrupted.csv')

# Merge for questions that need both datasets
merged = posts_clean.merge(users, on='user_id', how='left')

def separator(label):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}\n")

# ════════════════════════════════════════════════════════════════
#                        EASY LEVEL
# ════════════════════════════════════════════════════════════════

# ── E1: Platform Popularity ──
separator("E1 — Platform Popularity")
e1 = posts_clean.dropna(subset=['platform']).groupby('platform').size().reset_index(name='num_posts')
e1 = e1.sort_values('num_posts', ascending=False)
print(e1.to_string(index=False))
top_platform = e1.iloc[0]
print(f"\n>> Most popular platform: {top_platform['platform']} with {top_platform['num_posts']} posts")

# ── E2: Most Engaged Posts (Top 10) ──
separator("E2 — Most Engaged Posts (Top 10)")
e2 = posts_clean.dropna(subset=['likes']).copy()
e2['total_engagement'] = e2['likes'] + e2['shares'] + e2['comments']
e2_top = e2.nlargest(10, 'total_engagement')[['post_id', 'platform', 'likes', 'shares', 'comments', 'total_engagement']]
print(e2_top.to_string(index=False))

# ── E3: Average Engagement by Platform ──
separator("E3 — Average Engagement by Platform")
e3 = posts_clean.dropna(subset=['platform']).copy()
e3_stats = e3.groupby('platform').agg(
    avg_likes=('likes', 'mean'),
    avg_shares=('shares', 'mean'),
    avg_comments=('comments', 'mean')
).round(2)
e3_stats['avg_total_engagement'] = (e3_stats['avg_likes'] + e3_stats['avg_shares'] + e3_stats['avg_comments']).round(2)
e3_stats = e3_stats.sort_values('avg_total_engagement', ascending=False)
print(e3_stats.to_string())
print(f"\n>> Highest average total engagement: {e3_stats.index[0]} ({e3_stats.iloc[0]['avg_total_engagement']})")

# ── E4: Highly Shared but Poorly Liked ──
separator("E4 — Highly Shared but Poorly Liked")
e4 = posts_clean[(posts_clean['shares'] > 1500) & (posts_clean['likes'] < 500)].copy()
e4_out = e4[['post_id', 'platform', 'likes', 'shares', 'comments']]
print(f"Found {len(e4_out)} posts with >1500 shares but <500 likes:\n")
print(e4_out.to_string(index=False))

# ── E5: Users With Large Audiences (>40,000 followers) ──
separator("E5 — Users With Large Audiences (>40,000 followers)")
e5 = users[users['follower_count'] > 40000][['user_id', 'location', 'language', 'follower_count']]
e5 = e5.sort_values('follower_count', ascending=False)
print(f"Found {len(e5)} users with >40,000 followers:\n")
print(e5.to_string(index=False))


# ════════════════════════════════════════════════════════════════
#                       MEDIUM LEVEL
# ════════════════════════════════════════════════════════════════

# ── M1: Which Locations Generate the Most Engagement? ──
separator("M1 — Which Locations Generate the Most Engagement?")
m1 = merged.copy()
m1['total_engagement'] = m1['likes'].fillna(0) + m1['shares'] + m1['comments']
m1_stats = m1.groupby('location').agg(
    num_posts=('post_id', 'count'),
    total_engagement=('total_engagement', 'sum')
).reset_index()
m1_stats = m1_stats.sort_values('total_engagement', ascending=False)
print(m1_stats.to_string(index=False))

# ── M2: Do High Follower Users Get More Engagement? ──
separator("M2 — Do High Follower Users Get More Engagement?")
m2 = merged.copy()
m2['total_engagement'] = m2['likes'].fillna(0) + m2['shares'] + m2['comments']
m2['follower_group'] = np.where(m2['follower_count'] >= 25000, 'High (≥25K)', 'Low (<25K)')
m2_compare = m2.groupby('follower_group')['total_engagement'].mean().round(2)
print(m2_compare.to_string())
if m2_compare.get('High (≥25K)', 0) > m2_compare.get('Low (<25K)', 0):
    print("\n>> High follower users DO get more engagement on average.")
else:
    print("\n>> Surprisingly, low follower users get comparable or more engagement.")

# ── M3: Most Active Users (Top 10) ──
separator("M3 — Most Active Users (Top 10)")
m3 = merged.copy()
m3['total_engagement'] = m3['likes'].fillna(0) + m3['shares'] + m3['comments']
m3_stats = m3.groupby('user_id').agg(
    follower_count=('follower_count', 'first'),
    location=('location', 'first'),
    post_count=('post_id', 'count'),
    total_engagement=('total_engagement', 'sum')
).reset_index()
m3_top = m3_stats.nlargest(10, 'post_count')[['user_id', 'follower_count', 'location', 'post_count', 'total_engagement']]
print(m3_top.to_string(index=False))

# ── M4: Platform Behaviour by High Follower Users ──
separator("M4 — Platform Behaviour by High Follower Users (≥30K followers)")
m4 = merged[merged['follower_count'] >= 30000].copy()
m4['total_engagement'] = m4['likes'].fillna(0) + m4['shares'] + m4['comments']
m4_platform = m4.dropna(subset=['platform']).groupby('platform')['total_engagement'].mean().round(2).sort_values(ascending=False)
print(m4_platform.to_string())
print(f"\n>> Best platform for high-follower users: {m4_platform.index[0]} (avg engagement: {m4_platform.iloc[0]})")

# ── M5: Detect Suspicious Engagement (Top 20) ──
separator("M5 — Detect Suspicious Engagement")
m5 = posts_clean.copy()
m5_suspicious = m5[m5['shares'] > (m5['likes'].fillna(0) + m5['comments'])].copy()
m5_suspicious['total_engagement'] = m5_suspicious['likes'].fillna(0) + m5_suspicious['shares'] + m5_suspicious['comments']
m5_top = m5_suspicious.nlargest(20, 'total_engagement')[['post_id', 'platform', 'likes', 'shares', 'comments', 'total_engagement']]
print(f"Found {len(m5_suspicious)} posts where shares > likes + comments.")
print(f"\nTop 20:\n")
print(m5_top.to_string(index=False))


# ════════════════════════════════════════════════════════════════
#                        HARD LEVEL
# ════════════════════════════════════════════════════════════════

# ── H1: Find Users With Abnormally High Engagement ──
separator("H1 — Users With Abnormally High Engagement")
h1 = merged.copy()
h1['total_engagement'] = h1['likes'].fillna(0) + h1['shares'] + h1['comments']
h1_user = h1.groupby('user_id').agg(
    location=('location', 'first'),
    follower_count=('follower_count', 'first'),
    post_count=('post_id', 'count'),
    avg_engagement=('total_engagement', 'mean')
).reset_index()
overall_avg = h1['total_engagement'].mean()
h1_abnormal = h1_user[h1_user['avg_engagement'] > 2 * overall_avg].copy()
h1_abnormal['avg_engagement'] = h1_abnormal['avg_engagement'].round(2)
h1_abnormal = h1_abnormal.sort_values('avg_engagement', ascending=False)
print(f"Overall average engagement per post: {overall_avg:.2f}")
print(f"Threshold (2x): {2 * overall_avg:.2f}")
print(f"\nFound {len(h1_abnormal)} users with avg engagement > 2x overall average:\n")
print(h1_abnormal.to_string(index=False))

# ── H2: Rank Users Within Their Location (Top 3 per location) ──
separator("H2 — Rank Users Within Their Location (Top 3 per Location)")
h2 = merged.copy()
h2['total_engagement'] = h2['likes'].fillna(0) + h2['shares'] + h2['comments']
h2_user = h2.groupby(['user_id', 'location']).agg(
    total_engagement=('total_engagement', 'sum')
).reset_index()
h2_user['rank'] = h2_user.groupby('location')['total_engagement'].rank(method='dense', ascending=False)
h2_top3 = h2_user[h2_user['rank'] <= 3].sort_values(['location', 'rank'])
print(h2_top3.to_string(index=False))

# ── H3: Platform Performance Compared With Its Own Average ──
separator("H3 — Exceptional Posts (≥2x Platform Average)")
h3 = posts_clean.dropna(subset=['platform']).copy()
h3['total_engagement'] = h3['likes'].fillna(0) + h3['shares'] + h3['comments']
platform_avg = h3.groupby('platform')['total_engagement'].mean()
print("Platform averages:")
print(platform_avg.round(2).to_string())

h3 = h3.merge(platform_avg.rename('platform_avg'), on='platform')
h3_exceptional = h3[h3['total_engagement'] >= 2 * h3['platform_avg']].copy()
h3_summary = h3_exceptional.groupby('platform').size().reset_index(name='exceptional_posts')
print(f"\nExceptional posts per platform:")
print(h3_summary.to_string(index=False))
print(f"\nTotal exceptional posts: {len(h3_exceptional)}")
print(f"\nSample exceptional posts:")
print(h3_exceptional.nlargest(10, 'total_engagement')[['post_id', 'platform', 'total_engagement', 'platform_avg']].to_string(index=False))

# ── H4: Follower to Engagement Anomaly ──
separator("H4 — Follower to Engagement Anomaly")
h4 = merged.copy()
h4['total_engagement'] = h4['likes'].fillna(0) + h4['shares'] + h4['comments']
h4_user = h4.groupby('user_id').agg(
    follower_count=('follower_count', 'first'),
    location=('location', 'first'),
    total_engagement=('total_engagement', 'sum'),
    post_count=('post_id', 'count')
).reset_index()

top10_threshold = h4_user['total_engagement'].quantile(0.90)
h4_anomaly = h4_user[(h4_user['follower_count'] < 5000) & (h4_user['total_engagement'] >= top10_threshold)]
h4_anomaly = h4_anomaly.sort_values('total_engagement', ascending=False)
print(f"Top 10% engagement threshold: {top10_threshold:.0f}")
print(f"\nFound {len(h4_anomaly)} users with <5K followers but top-10% engagement:\n")
print(h4_anomaly.to_string(index=False))

# ── H5: Identify Data Anomalies (uses CORRUPTED dataset) ──
separator("H5 — Identify Data Anomalies (from Corrupted Dataset)")
import re

anomalies = []

for _, row in posts_corrupted.iterrows():
    issues = []
    
    # Check for negative likes
    try:
        likes_val = pd.to_numeric(row.get('likes', None), errors='coerce')
        if pd.notna(likes_val) and likes_val < 0:
            issues.append('Negative likes')
    except:
        pass
    
    # Check for missing platform
    if pd.isna(row.get('platform', None)) or str(row.get('platform', '')).strip() == '':
        issues.append('Missing platform')
    
    # Check for missing text content
    if pd.isna(row.get('text_content', None)) or str(row.get('text_content', '')).strip() == '':
        issues.append('Missing text content')
    
    # Check for HTML entities/tags
    text = str(row.get('text_content', ''))
    if re.search(r'&amp;|&lt;|&gt;|&nbsp;|<div>|<br>|</div>|<br/>|<span>|</span>|<p>|</p>', text, re.IGNORECASE):
        issues.append('Contains HTML tags/entities')
    
    if issues:
        anomalies.append({
            'post_id': row['post_id'],
            'anomaly_type': ' | '.join(issues)
        })

h5_df = pd.DataFrame(anomalies)
print(f"Total anomalous posts found: {len(h5_df)}\n")

# Summary by anomaly type
for anomaly_type in ['Negative likes', 'Missing platform', 'Missing text content', 'Contains HTML tags/entities']:
    count = h5_df['anomaly_type'].str.contains(anomaly_type).sum()
    print(f"  {anomaly_type}: {count} posts")

print(f"\nSample anomalous posts (first 20):")
print(h5_df.head(20).to_string(index=False))

# ── H6: Most Suspicious High Impact Users ──
separator("H6 — Most Suspicious High Impact Users")
h6 = merged.copy()
h6['total_engagement'] = h6['likes'].fillna(0) + h6['shares'] + h6['comments']

# User-level aggregation
h6_user = h6.groupby('user_id').agg(
    location=('location', 'first'),
    follower_count=('follower_count', 'first'),
    post_count=('post_id', 'count'),
    total_engagement=('total_engagement', 'sum'),
    avg_engagement=('total_engagement', 'mean')
).reset_index()

# Condition 1: fewer than 10,000 followers
cond1 = h6_user['follower_count'] < 10000

# Condition 2: avg post engagement > overall average
overall_avg_eng = h6['total_engagement'].mean()
cond2 = h6_user['avg_engagement'] > overall_avg_eng

# Condition 3: at least one post with more shares than likes
shares_gt_likes = h6.groupby('user_id').apply(
    lambda x: (x['shares'] > x['likes'].fillna(0)).any()
).reset_index(name='has_shares_gt_likes')
h6_user = h6_user.merge(shares_gt_likes, on='user_id')
cond3 = h6_user['has_shares_gt_likes']

h6_suspicious = h6_user[cond1 & cond2 & cond3].copy()
h6_suspicious['avg_engagement'] = h6_suspicious['avg_engagement'].round(2)
h6_suspicious = h6_suspicious.sort_values('total_engagement', ascending=False)
h6_out = h6_suspicious[['user_id', 'location', 'follower_count', 'post_count', 'avg_engagement', 'total_engagement']]

print(f"Overall average engagement: {overall_avg_eng:.2f}")
print(f"\nFound {len(h6_suspicious)} suspicious high-impact users:\n")
print(h6_out.to_string(index=False))


# ════════════════════════════════════════════════════════════════
#                     SAVE ALL RESULTS
# ════════════════════════════════════════════════════════════════
separator("SAVING RESULTS")

results = {
    'E1_platform_popularity': e1,
    'E2_most_engaged_posts': e2_top,
    'E3_avg_engagement_by_platform': e3_stats.reset_index(),
    'E4_high_shares_low_likes': e4_out,
    'E5_large_audience_users': e5,
    'M1_location_engagement': m1_stats,
    'M3_most_active_users': m3_top,
    'M5_suspicious_engagement': m5_top,
    'H1_abnormal_engagement_users': h1_abnormal,
    'H2_top3_per_location': h2_top3,
    'H4_follower_anomaly': h4_anomaly,
    'H5_data_anomalies': h5_df,
    'H6_suspicious_users': h6_out
}

with pd.ExcelWriter('Phase2_Answers.xlsx', engine='openpyxl') as writer:
    for sheet_name, df in results.items():
        df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

print("All results saved to Phase2_Answers.xlsx!")
print("\n" + "="*70)
print("  PHASE 2 COMPLETE — ALL 16 QUESTIONS ANSWERED")
print("="*70)
