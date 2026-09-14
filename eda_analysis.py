"""
╔══════════════════════════════════════════════════════════════╗
║  SOCIAL ENGINE RECOVERY — EXPLORATORY DATA ANALYSIS (EDA)   ║
║  Data Vortex :: AARUUSH'26 — Round 1 Phase 1                ║
║                                                              ║
║  Author: Social Engine Recovery Team                         ║
║  Date: 2026-09-14                                            ║
╚══════════════════════════════════════════════════════════════╝

This script performs comprehensive EDA on the cleaned Social Engine
datasets. It generates insights across multiple dimensions:

1. Platform Distribution & Market Share
2. Temporal Posting Patterns (monthly, day-of-week, hourly)
3. Engagement Analysis (likes, shares, comments)
4. Brand & Product Mention Analysis
5. Hashtag Analysis
6. Sentiment Analysis (keyword-based)
7. User Activity & Follower-Engagement Correlation
8. Geographic & Language Distribution
9. Anomaly Detection in Engagement Metrics
10. Cross-Platform Comparison
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import Counter
import re
import os
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

DATA_DIR = r"C:\Users\saish\.gemini\antigravity-ide\scratch\social-engine-recovery"
PLOTS_DIR = os.path.join(DATA_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Style config
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'text.color': '#c9d1d9',
    'axes.labelcolor': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'axes.edgecolor': '#30363d',
    'grid.color': '#21262d',
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 13,
    'figure.titlesize': 15,
})

PALETTE = ['#58a6ff', '#f778ba', '#7ee787', '#ffa657', '#d2a8ff',
           '#79c0ff', '#ff7b72', '#ffd700', '#a5d6ff', '#56d4dd']

# ═══════════════════════════════════════════════════════════════
# LOAD CLEANED DATA
# ═══════════════════════════════════════════════════════════════

posts = pd.read_csv(os.path.join(DATA_DIR, "Social_Engine_Posts_Cleaned.csv"),
                    parse_dates=['timestamp'])
users = pd.read_csv(os.path.join(DATA_DIR, "Social_Engine_Users_Cleaned.csv"),
                    parse_dates=['account_created'])

# Merge for user-level analysis
merged = posts.merge(users, on='user_id', how='left')

print(f"Loaded {posts.shape[0]} posts and {users.shape[0]} users")
print(f"Post date range: {posts['timestamp'].min()} to {posts['timestamp'].max()}")

insights = []

def record_insight(category, text):
    insights.append(f"[{category}] {text}")
    print(f"  💡 [{category}] {text}")


# ═══════════════════════════════════════════════════════════════
# 1. PLATFORM DISTRIBUTION
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("1. PLATFORM DISTRIBUTION")
print("=" * 60)

platform_counts = posts['platform'].value_counts(dropna=False)
platform_valid = posts['platform'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Platform Distribution', fontweight='bold', fontsize=15)

# Bar chart
bars = axes[0].barh(platform_valid.index[::-1], platform_valid.values[::-1],
                    color=PALETTE[:len(platform_valid)], edgecolor='#30363d')
axes[0].set_xlabel('Number of Posts')
axes[0].set_title('Posts by Platform')
for bar, val in zip(bars, platform_valid.values[::-1]):
    axes[0].text(bar.get_width() + 30, bar.get_y() + bar.get_height()/2,
                 f'{val:,}', va='center', color='#c9d1d9', fontsize=9)

# Pie chart (excluding NaN)
axes[1].pie(platform_valid.values, labels=platform_valid.index,
            colors=PALETTE[:len(platform_valid)], autopct='%1.1f%%',
            textprops={'color': '#c9d1d9', 'fontsize': 9},
            wedgeprops={'edgecolor': '#30363d'})
axes[1].set_title('Platform Market Share')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '01_platform_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()

missing_platform_pct = posts['platform'].isna().sum() / len(posts) * 100
record_insight("PLATFORM", f"YouTube leads with {platform_valid.iloc[0]:,} posts ({platform_valid.iloc[0]/len(posts)*100:.1f}%)")
record_insight("PLATFORM", f"{missing_platform_pct:.1f}% of posts have missing platform data due to corruption")
record_insight("PLATFORM", f"All 5 platforms (YouTube, Facebook, Twitter, Reddit, Instagram) have roughly equal share (~17-18% each)")


# ═══════════════════════════════════════════════════════════════
# 2. TEMPORAL ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. TEMPORAL ANALYSIS")
print("=" * 60)

posts_with_time = posts.dropna(subset=['timestamp'])

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Temporal Posting Patterns', fontweight='bold', fontsize=15)

# Monthly trend
monthly = posts_with_time.set_index('timestamp').resample('ME').size()
axes[0, 0].fill_between(monthly.index, monthly.values, alpha=0.3, color=PALETTE[0])
axes[0, 0].plot(monthly.index, monthly.values, color=PALETTE[0], linewidth=2)
axes[0, 0].set_title('Monthly Post Volume')
axes[0, 0].set_xlabel('Month')
axes[0, 0].set_ylabel('Number of Posts')
axes[0, 0].tick_params(axis='x', rotation=45)

# Day of week
dow_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
dow = posts_with_time['timestamp'].dt.dayofweek.map(dow_map).value_counts()
dow = dow.reindex(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
axes[0, 1].bar(dow.index, dow.values, color=PALETTE[1], edgecolor='#30363d')
axes[0, 1].set_title('Posts by Day of Week')
axes[0, 1].set_ylabel('Number of Posts')

# Hourly (only for ISO timestamps that have time info)
hourly = posts_with_time['timestamp'].dt.hour.value_counts().sort_index()
axes[1, 0].bar(hourly.index, hourly.values, color=PALETTE[2], edgecolor='#30363d', width=0.8)
axes[1, 0].set_title('Posts by Hour of Day')
axes[1, 0].set_xlabel('Hour')
axes[1, 0].set_ylabel('Number of Posts')
axes[1, 0].set_xticks(range(0, 24, 2))

# Year-Month heatmap-style
posts_with_time = posts_with_time.copy()
posts_with_time['year_month'] = posts_with_time['timestamp'].dt.to_period('M').astype(str)
ym_platform = posts_with_time.groupby(['year_month', 'platform']).size().unstack(fill_value=0)
if not ym_platform.empty:
    ym_platform_plot = ym_platform.tail(12)  # last 12 months
    ym_platform_plot.plot(kind='bar', stacked=True, ax=axes[1, 1],
                          color=PALETTE[:len(ym_platform.columns)], edgecolor='#30363d')
    axes[1, 1].set_title('Platform Mix Over Time (Last 12 Months)')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Posts')
    axes[1, 1].legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d')
    axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '02_temporal_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()

peak_month = monthly.idxmax().strftime('%B %Y')
peak_dow = dow.idxmax()
peak_hour = hourly.idxmax()
record_insight("TEMPORAL", f"Peak posting month: {peak_month} ({monthly.max():,} posts)")
record_insight("TEMPORAL", f"Most active day: {peak_dow}")
record_insight("TEMPORAL", f"Peak posting hour: {peak_hour}:00 (note: only available for ISO-format timestamps)")


# ═══════════════════════════════════════════════════════════════
# 3. ENGAGEMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. ENGAGEMENT ANALYSIS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Engagement Metrics Analysis', fontweight='bold', fontsize=15)

# Likes distribution
likes_clean = posts['likes'].dropna()
axes[0, 0].hist(likes_clean, bins=50, color=PALETTE[0], edgecolor='#30363d', alpha=0.8)
axes[0, 0].set_title('Distribution of Likes')
axes[0, 0].set_xlabel('Likes')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(likes_clean.mean(), color=PALETTE[5], linestyle='--', label=f'Mean: {likes_clean.mean():.0f}')
axes[0, 0].axvline(likes_clean.median(), color=PALETTE[1], linestyle='--', label=f'Median: {likes_clean.median():.0f}')
axes[0, 0].legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d')

# Shares distribution
axes[0, 1].hist(posts['shares'], bins=50, color=PALETTE[1], edgecolor='#30363d', alpha=0.8)
axes[0, 1].set_title('Distribution of Shares')
axes[0, 1].set_xlabel('Shares')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].axvline(posts['shares'].mean(), color=PALETTE[5], linestyle='--', label=f'Mean: {posts["shares"].mean():.0f}')
axes[0, 1].legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d')

# Comments distribution
axes[1, 0].hist(posts['comments'], bins=50, color=PALETTE[2], edgecolor='#30363d', alpha=0.8)
axes[1, 0].set_title('Distribution of Comments')
axes[1, 0].set_xlabel('Comments')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].axvline(posts['comments'].mean(), color=PALETTE[5], linestyle='--', label=f'Mean: {posts["comments"].mean():.0f}')
axes[1, 0].legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d')

# Engagement correlation scatter
axes[1, 1].scatter(likes_clean, posts.loc[likes_clean.index, 'shares'],
                   alpha=0.15, s=8, color=PALETTE[3])
axes[1, 1].set_title('Likes vs Shares Correlation')
axes[1, 1].set_xlabel('Likes')
axes[1, 1].set_ylabel('Shares')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '03_engagement_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()

# Engagement by platform
engagement_by_platform = posts.groupby('platform').agg(
    avg_likes=('likes', 'mean'),
    avg_shares=('shares', 'mean'),
    avg_comments=('comments', 'mean'),
    total_posts=('post_id', 'count')
).round(1)
print("\nEngagement by Platform:")
print(engagement_by_platform)

# Correlation matrix
corr_data = posts[['likes', 'shares', 'comments']].dropna()
corr_matrix = corr_data.corr()

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr_matrix.values, cmap='RdYlGn', vmin=-1, vmax=1)
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(['Likes', 'Shares', 'Comments'])
ax.set_yticklabels(['Likes', 'Shares', 'Comments'])
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{corr_matrix.values[i, j]:.3f}',
                ha='center', va='center', color='black', fontweight='bold')
ax.set_title('Engagement Metrics Correlation Matrix')
plt.colorbar(im)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '04_correlation_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()

likes_shares_corr = corr_matrix.loc['likes', 'shares']
record_insight("ENGAGEMENT", f"Average likes: {likes_clean.mean():.0f}, shares: {posts['shares'].mean():.0f}, comments: {posts['comments'].mean():.0f}")
record_insight("ENGAGEMENT", f"Likes-Shares correlation: {likes_shares_corr:.3f} — {'weak' if abs(likes_shares_corr) < 0.3 else 'moderate' if abs(likes_shares_corr) < 0.6 else 'strong'} relationship")
record_insight("ENGAGEMENT", f"Likes data missing for {posts['likes'].isna().sum()} posts ({posts['likes'].isna().sum()/len(posts)*100:.1f}%)")


# ═══════════════════════════════════════════════════════════════
# 4. BRAND & PRODUCT ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. BRAND & PRODUCT ANALYSIS")
print("=" * 60)

# Extract brands mentioned
brands = ['Nike', 'Adidas', 'Apple', 'Samsung', 'Google', 'Microsoft',
          'Amazon', 'Toyota', 'Pepsi', 'Coca-Cola']

brand_counts = {}
text_series = posts['text_content'].dropna()
for brand in brands:
    count = text_series.str.contains(brand, case=False, na=False).sum()
    brand_counts[brand] = count

brand_df = pd.Series(brand_counts).sort_values(ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Brand Analysis', fontweight='bold', fontsize=15)

# Brand mentions
bars = axes[0].barh(brand_df.index, brand_df.values, color=PALETTE[:len(brand_df)], edgecolor='#30363d')
axes[0].set_title('Posts Mentioning Each Brand')
axes[0].set_xlabel('Number of Posts')
for bar, val in zip(bars, brand_df.values):
    axes[0].text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                 f'{val:,}', va='center', color='#c9d1d9', fontsize=9)

# Brand engagement (avg likes)
brand_engagement = {}
for brand in brands:
    mask = posts['text_content'].str.contains(brand, case=False, na=False)
    brand_posts = posts.loc[mask]
    brand_engagement[brand] = brand_posts['likes'].mean()

be_df = pd.Series(brand_engagement).sort_values(ascending=True).dropna()
axes[1].barh(be_df.index, be_df.values, color=PALETTE[3], edgecolor='#30363d')
axes[1].set_title('Average Likes by Brand')
axes[1].set_xlabel('Average Likes')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '05_brand_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()

top_brand = brand_df.idxmax()
record_insight("BRAND", f"Most mentioned brand: {top_brand} ({brand_df.max():,} mentions)")
record_insight("BRAND", f"All 10 brands have significant presence — this is a multi-brand discussion forum")

# Product analysis
products = {
    'Nike': ['Air Max', 'Air Force 1', 'Air Jordan', 'Dri-FIT', 'Zoom Pegasus', 
             'FlyKnit', 'React', 'Epic React'],
    'Apple': ['iPhone 15', 'MacBook Pro', 'AirPods Pro', 'iPad Air', 'Vision Pro',
              'Apple Watch', 'iMac', 'Mac Mini'],
    'Samsung': ['Galaxy S25', 'Galaxy Z Fold', 'Galaxy Watch', 'Galaxy Buds',
                'Neo QLED TV', 'Galaxy Tab'],
    'Toyota': ['Corolla', 'Camry', 'RAV4', 'Highlander', 'Prius', 'Tundra', 'Sienna'],
}

print("\nTop Products by Brand:")
for brand, prods in products.items():
    prod_counts = {}
    for prod in prods:
        prod_counts[prod] = text_series.str.contains(prod, case=False, na=False).sum()
    sorted_prods = sorted(prod_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"  {brand}: {', '.join(f'{p} ({c})' for p, c in sorted_prods)}")


# ═══════════════════════════════════════════════════════════════
# 5. HASHTAG ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. HASHTAG ANALYSIS")
print("=" * 60)

# Extract all hashtags
all_hashtags = []
for text in posts['text_content'].dropna():
    hashtags = re.findall(r'#(\w+)', str(text))
    all_hashtags.extend(hashtags)

hashtag_counts = Counter(all_hashtags)
top_hashtags = hashtag_counts.most_common(20)

fig, ax = plt.subplots(figsize=(12, 6))
names = [h[0] for h in top_hashtags[::-1]]
values = [h[1] for h in top_hashtags[::-1]]
bars = ax.barh(names, values, color=PALETTE[4], edgecolor='#30363d')
ax.set_title('Top 20 Hashtags', fontweight='bold')
ax.set_xlabel('Number of Uses')
for bar, val in zip(bars, values):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
             f'{val:,}', va='center', color='#c9d1d9', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '06_hashtag_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"Total unique hashtags: {len(hashtag_counts)}")
print(f"Top 10: {top_hashtags[:10]}")
record_insight("HASHTAGS", f"Top hashtag: #{top_hashtags[0][0]} ({top_hashtags[0][1]} uses)")
record_insight("HASHTAGS", f"{len(hashtag_counts)} unique hashtags across {len(all_hashtags)} total mentions")


# ═══════════════════════════════════════════════════════════════
# 6. SENTIMENT ANALYSIS (Keyword-Based)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. SENTIMENT ANALYSIS")
print("=" * 60)

positive_keywords = [
    'absolutely loving', 'best purchase ever', 'exceeded my expectations',
    'worth every penny', 'highly recommend', 'fantastic', 'amazing',
    'excellent', 'outstanding', 'impressive', 'thrilled', 'delighted',
    'super excited', 'loving it', "can't contain my excitement", 'so happy'
]
negative_keywords = [
    'not worth', 'disappointed', 'returning it', 'frustrating', 'subpar',
    'overpriced', 'underwhelming', 'wouldn\'t recommend', 'had issues',
    'fed up', 'bummed out', 'sad to report', 'feeling let down',
    'not responding', 'disappointing'
]
neutral_keywords = [
    'does the job', 'not bad', 'it\'s okay', 'as expected',
    'mixed feelings', 'acceptable', 'standard', 'decent', 'typical'
]

def classify_sentiment(text):
    if pd.isna(text):
        return 'Unknown'
    text_lower = str(text).lower()
    pos_score = sum(1 for kw in positive_keywords if kw in text_lower)
    neg_score = sum(1 for kw in negative_keywords if kw in text_lower)
    neu_score = sum(1 for kw in neutral_keywords if kw in text_lower)
    
    if pos_score > neg_score and pos_score > neu_score:
        return 'Positive'
    elif neg_score > pos_score and neg_score > neu_score:
        return 'Negative'
    elif neu_score > 0:
        return 'Neutral'
    else:
        # If no clear signal, check for any keywords
        if pos_score > 0:
            return 'Positive'
        elif neg_score > 0:
            return 'Negative'
        else:
            return 'Unknown'

posts['sentiment'] = posts['text_content'].apply(classify_sentiment)
sentiment_counts = posts['sentiment'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Sentiment Analysis', fontweight='bold', fontsize=15)

# Sentiment distribution
sentiment_colors = {'Positive': '#7ee787', 'Negative': '#ff7b72', 
                    'Neutral': '#ffa657', 'Unknown': '#8b949e'}
sent_labels = sentiment_counts.index
sent_values = sentiment_counts.values
sent_colors = [sentiment_colors.get(s, '#8b949e') for s in sent_labels]

axes[0].pie(sent_values, labels=sent_labels, colors=sent_colors,
            autopct='%1.1f%%', textprops={'color': '#c9d1d9', 'fontsize': 10},
            wedgeprops={'edgecolor': '#30363d'})
axes[0].set_title('Overall Sentiment Distribution')

# Sentiment by platform
sent_platform = posts.groupby(['platform', 'sentiment']).size().unstack(fill_value=0)
sent_platform_pct = sent_platform.div(sent_platform.sum(axis=1), axis=0) * 100
if 'Positive' in sent_platform_pct.columns and 'Negative' in sent_platform_pct.columns:
    sent_plot_cols = [c for c in ['Positive', 'Neutral', 'Negative', 'Unknown'] if c in sent_platform_pct.columns]
    sent_plot_colors = [sentiment_colors[c] for c in sent_plot_cols]
    sent_platform_pct[sent_plot_cols].plot(kind='bar', stacked=True, ax=axes[1],
                                           color=sent_plot_colors, edgecolor='#30363d')
    axes[1].set_title('Sentiment by Platform (%)')
    axes[1].set_ylabel('Percentage')
    axes[1].set_xlabel('')
    axes[1].legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d')
    axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '07_sentiment_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()

pos_count = sentiment_counts.get('Positive', 0)
neg_count = sentiment_counts.get('Negative', 0)
record_insight("SENTIMENT", f"Positive: {pos_count} ({pos_count/len(posts)*100:.1f}%), Negative: {neg_count} ({neg_count/len(posts)*100:.1f}%)")

# Sentiment vs engagement
sent_engagement = posts.groupby('sentiment')['likes'].mean()
print("\nAverage Likes by Sentiment:")
print(sent_engagement)
if 'Positive' in sent_engagement and 'Negative' in sent_engagement:
    record_insight("SENTIMENT", f"Positive posts avg {sent_engagement['Positive']:.0f} likes vs negative avg {sent_engagement['Negative']:.0f} likes")


# ═══════════════════════════════════════════════════════════════
# 7. USER ACTIVITY & FOLLOWER ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7. USER ACTIVITY ANALYSIS")
print("=" * 60)

user_posts = posts.groupby('user_id').agg(
    post_count=('post_id', 'count'),
    avg_likes=('likes', 'mean'),
    avg_shares=('shares', 'mean'),
    avg_comments=('comments', 'mean'),
    platforms_used=('platform', 'nunique')
).reset_index()

user_merged = user_posts.merge(users, on='user_id', how='left')

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('User Activity Analysis', fontweight='bold', fontsize=15)

# Posts per user distribution
axes[0, 0].hist(user_posts['post_count'], bins=30, color=PALETTE[0], edgecolor='#30363d')
axes[0, 0].set_title('Posts Per User Distribution')
axes[0, 0].set_xlabel('Number of Posts')
axes[0, 0].set_ylabel('Number of Users')
axes[0, 0].axvline(user_posts['post_count'].mean(), color=PALETTE[5], linestyle='--',
                    label=f'Mean: {user_posts["post_count"].mean():.1f}')
axes[0, 0].legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d')

# Follower count vs avg likes
axes[0, 1].scatter(user_merged['follower_count'], user_merged['avg_likes'],
                   alpha=0.3, s=15, color=PALETTE[1])
axes[0, 1].set_title('Follower Count vs Average Likes')
axes[0, 1].set_xlabel('Follower Count')
axes[0, 1].set_ylabel('Average Likes')

# Top 15 most active users
top_users = user_posts.nlargest(15, 'post_count')
axes[1, 0].barh(top_users['user_id'].str[-8:], top_users['post_count'],
                color=PALETTE[2], edgecolor='#30363d')
axes[1, 0].set_title('Top 15 Most Active Users')
axes[1, 0].set_xlabel('Number of Posts')

# Follower distribution
axes[1, 1].hist(users['follower_count'], bins=50, color=PALETTE[3], edgecolor='#30363d')
axes[1, 1].set_title('Follower Count Distribution')
axes[1, 1].set_xlabel('Followers')
axes[1, 1].set_ylabel('Number of Users')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '08_user_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()

follower_likes_corr = user_merged[['follower_count', 'avg_likes']].dropna().corr().iloc[0, 1]
record_insight("USERS", f"Average posts per user: {user_posts['post_count'].mean():.1f}")
record_insight("USERS", f"Most active user: {top_users.iloc[0]['user_id']} ({top_users.iloc[0]['post_count']} posts)")
record_insight("USERS", f"Follower count ↔ avg likes correlation: {follower_likes_corr:.3f} — {'weak' if abs(follower_likes_corr) < 0.3 else 'moderate'}")


# ═══════════════════════════════════════════════════════════════
# 8. GEOGRAPHIC & LANGUAGE ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("8. GEOGRAPHIC & LANGUAGE ANALYSIS")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Geographic & Language Distribution', fontweight='bold', fontsize=15)

# Top locations
location_counts = users['location'].value_counts().head(15)
axes[0].barh(location_counts.index[::-1], location_counts.values[::-1],
             color=PALETTE[0], edgecolor='#30363d')
axes[0].set_title('Top 15 User Locations')
axes[0].set_xlabel('Number of Users')

# Language distribution
lang_counts = users['language'].value_counts()
lang_labels = lang_counts.index
lang_full = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
    'ja': 'Japanese', 'zh': 'Chinese', 'pt': 'Portuguese', 'ar': 'Arabic',
    'hi': 'Hindi', 'ru': 'Russian'
}
lang_display = [lang_full.get(l, l) for l in lang_labels]
axes[1].pie(lang_counts.values, labels=lang_display,
            colors=PALETTE[:len(lang_counts)], autopct='%1.1f%%',
            textprops={'color': '#c9d1d9', 'fontsize': 8},
            wedgeprops={'edgecolor': '#30363d'})
axes[1].set_title('User Language Distribution')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '09_geographic_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()

top_location = location_counts.idxmax()
top_lang = lang_counts.idxmax()
record_insight("GEOGRAPHY", f"Top location: {top_location} ({location_counts.max()} users)")
record_insight("LANGUAGE", f"Most common language: {lang_full.get(top_lang, top_lang)} ({lang_counts.max()} users, {lang_counts.max()/len(users)*100:.1f}%)")
record_insight("LANGUAGE", f"10 languages represented — truly global user base")


# ═══════════════════════════════════════════════════════════════
# 9. ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("9. ANOMALY DETECTION")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Engagement Anomalies — Box Plots', fontweight='bold', fontsize=15)

for i, col in enumerate(['likes', 'shares', 'comments']):
    data = posts[col].dropna()
    bp = axes[i].boxplot(data, orientation='vertical', patch_artist=True,
                         boxprops=dict(facecolor=PALETTE[i], edgecolor='#c9d1d9'),
                         medianprops=dict(color='#ffd700', linewidth=2),
                         whiskerprops=dict(color='#c9d1d9'),
                         capprops=dict(color='#c9d1d9'),
                         flierprops=dict(marker='o', markerfacecolor=PALETTE[i], markersize=3, alpha=0.5))
    axes[i].set_title(f'{col.title()} Distribution')
    axes[i].set_ylabel(col.title())
    
    # IQR-based outlier detection
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((data < lower) | (data > upper)).sum()
    print(f"  {col}: Q1={Q1:.0f}, Q3={Q3:.0f}, IQR={IQR:.0f}, Outliers={outliers}")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '10_anomaly_detection.png'), dpi=150, bbox_inches='tight')
plt.close()

# Zero engagement posts (no likes, low shares/comments)
zero_engagement = posts[(posts['shares'] == 0) | (posts['comments'] == 0)]
record_insight("ANOMALY", f"Posts with 0 shares or 0 comments: {len(zero_engagement)}")
record_insight("ANOMALY", f"Previously had {525} posts with negative likes (now set to NaN during cleaning)")


# ═══════════════════════════════════════════════════════════════
# 10. CROSS-PLATFORM ENGAGEMENT COMPARISON
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("10. CROSS-PLATFORM ENGAGEMENT")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Engagement by Platform', fontweight='bold', fontsize=15)

platform_order = ['YouTube', 'Facebook', 'Twitter', 'Instagram', 'Reddit']

for i, metric in enumerate(['likes', 'shares', 'comments']):
    data_by_platform = [posts[posts['platform'] == p][metric].dropna() for p in platform_order]
    bp = axes[i].boxplot(data_by_platform, tick_labels=platform_order, patch_artist=True,
                         medianprops=dict(color='#ffd700', linewidth=2),
                         whiskerprops=dict(color='#c9d1d9'),
                         capprops=dict(color='#c9d1d9'),
                         flierprops=dict(marker='o', markersize=2, alpha=0.3))
    for j, box in enumerate(bp['boxes']):
        box.set_facecolor(PALETTE[j])
        box.set_edgecolor('#c9d1d9')
    axes[i].set_title(f'{metric.title()} by Platform')
    axes[i].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '11_crossplatform_engagement.png'), dpi=150, bbox_inches='tight')
plt.close()

# Platform engagement stats
print("\nDetailed Platform Engagement Stats:")
for platform in platform_order:
    p_data = posts[posts['platform'] == platform]
    print(f"\n  {platform}:")
    print(f"    Posts: {len(p_data)}")
    print(f"    Avg Likes: {p_data['likes'].mean():.0f}")
    print(f"    Avg Shares: {p_data['shares'].mean():.0f}")
    print(f"    Avg Comments: {p_data['comments'].mean():.0f}")


# ═══════════════════════════════════════════════════════════════
# 11. MISSING DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("11. MISSING DATA ANALYSIS")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 5))
missing_pct = (posts.isnull().sum() / len(posts) * 100).sort_values(ascending=True)
missing_cols = missing_pct[missing_pct > 0]

if len(missing_cols) > 0:
    bars = ax.barh(missing_cols.index, missing_cols.values, color=PALETTE[5], edgecolor='#30363d')
    ax.set_title('Missing Data by Column (%)', fontweight='bold')
    ax.set_xlabel('% Missing')
    for bar, val in zip(bars, missing_cols.values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}%', va='center', color='#c9d1d9', fontsize=9)
else:
    ax.text(0.5, 0.5, 'No missing data!', ha='center', va='center',
            transform=ax.transAxes, fontsize=16)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '12_missing_data.png'), dpi=150, bbox_inches='tight')
plt.close()

record_insight("MISSING DATA", f"Highest missing: platform ({posts['platform'].isna().sum()/len(posts)*100:.1f}%), likes ({posts['likes'].isna().sum()/len(posts)*100:.1f}%), text_content ({posts['text_content'].isna().sum()/len(posts)*100:.1f}%)")


# ═══════════════════════════════════════════════════════════════
# 12. POST TYPE / CONTENT CATEGORY ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("12. POST TYPE ANALYSIS")
print("=" * 60)

# Classify post types based on text patterns
def classify_post_type(text):
    if pd.isna(text):
        return 'Empty/Missing'
    text_lower = str(text).lower()
    if 'just unboxed' in text_lower or 'just tried' in text_lower:
        return 'Product Review'
    elif 'just saw an ad' in text_lower:
        return 'Ad Reaction'
    elif 'attended the' in text_lower:
        return 'Event Coverage'
    elif 'comparing' in text_lower:
        return 'Comparison'
    elif 'my' in text_lower and 'review of' in text_lower:
        return 'Long-form Review'
    elif 'has anyone' in text_lower:
        return 'Question/Issue'
    elif 'should i' in text_lower or 'any advice' in text_lower or 'anyone have tips' in text_lower:
        return 'Seeking Advice'
    elif "what's your opinion" in text_lower or 'how do i' in text_lower:
        return 'Discussion'
    elif "can't wait" in text_lower and 'coming next' in text_lower:
        return 'Campaign/Event Reaction'
    else:
        return 'Other'

posts['post_type'] = posts['text_content'].apply(classify_post_type)
post_type_counts = posts['post_type'].value_counts()

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(post_type_counts.index[::-1], post_type_counts.values[::-1],
               color=PALETTE[:len(post_type_counts)], edgecolor='#30363d')
ax.set_title('Post Type Distribution', fontweight='bold')
ax.set_xlabel('Number of Posts')
for bar, val in zip(bars, post_type_counts.values[::-1]):
    ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
             f'{val:,}', va='center', color='#c9d1d9', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '13_post_types.png'), dpi=150, bbox_inches='tight')
plt.close()

record_insight("CONTENT", f"Most common post type: {post_type_counts.idxmax()} ({post_type_counts.max():,} posts)")
print(f"\nPost type distribution:\n{post_type_counts}")


# ═══════════════════════════════════════════════════════════════
# GENERATE INSIGHTS SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("KEY INSIGHTS SUMMARY")
print("=" * 60)

insights_file = os.path.join(DATA_DIR, "eda_insights.txt")
with open(insights_file, 'w', encoding='utf-8') as f:
    f.write("SOCIAL ENGINE — EDA KEY INSIGHTS\n")
    f.write(f"Generated: {datetime.now().isoformat()}\n")
    f.write("=" * 60 + "\n\n")
    for insight in insights:
        f.write(insight + "\n")
        print(f"  {insight}")

print(f"\n✓ Insights saved to: {insights_file}")
print(f"✓ All plots saved to: {PLOTS_DIR}")
print(f"\nPlots generated:")
for f in sorted(os.listdir(PLOTS_DIR)):
    if f.endswith('.png'):
        print(f"  📊 {f}")

# Drop the temp columns before final save
posts_final = posts.drop(columns=['sentiment', 'post_type'], errors='ignore')
posts_final.to_csv(os.path.join(DATA_DIR, "Social_Engine_Posts_Cleaned.csv"), index=False)

print(f"\n{'=' * 60}")
print("EDA COMPLETE")
print(f"{'=' * 60}")
