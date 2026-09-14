"""
╔══════════════════════════════════════════════════════════════╗
║  SOCIAL ENGINE RECOVERY — ADVANCED ANALYTICS                 ║
║  Data Vortex :: AARUUSH'26 — Round 1 Phase 1                 ║
║                                                              ║
║  This script goes beyond basic EDA.                          ║
║  Network analysis, NLP topic modeling, time series           ║
║  decomposition, engagement prediction, and anomaly           ║
║  storytelling — the stuff that wins hackathons.              ║
╚══════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import re
import os
import warnings
warnings.filterwarnings('ignore')

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans
import networkx as nx

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

DATA_DIR = r"C:\Users\saish\.gemini\antigravity-ide\scratch\social-engine-recovery"
PLOTS_DIR = os.path.join(DATA_DIR, "plots")

# Dark theme to match our previous plots
plt.style.use('dark_background')
PALETTE = ['#58a6ff', '#3fb950', '#f97583', '#d2a8ff', '#ffa657',
           '#79c0ff', '#56d364', '#ff7b72', '#bc8cff', '#d29922']
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'grid.alpha': 0.5,
    'font.size': 11,
})

# ═══════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════

posts = pd.read_csv(os.path.join(DATA_DIR, "Social_Engine_Posts_Cleaned.csv"))
users = pd.read_csv(os.path.join(DATA_DIR, "Social_Engine_Users_Cleaned.csv"))
posts['timestamp'] = pd.to_datetime(posts['timestamp'])
users['account_created'] = pd.to_datetime(users['account_created'])

print(f"Loaded {len(posts)} posts and {len(users)} users")
print(f"Post date range: {posts['timestamp'].min()} to {posts['timestamp'].max()}")

advanced_insights = []

def log_insight(category, insight):
    print(f"  >> [{category}] {insight}")
    advanced_insights.append(f"[{category}] {insight}")


# ═══════════════════════════════════════════════════════════════
# 1. NLP TOPIC MODELING (LDA)
#    Discover hidden discussion themes in the text content
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("1. NLP TOPIC MODELING")
print(f"{'='*60}")

text_data = posts['text_content'].dropna()
text_data = text_data[text_data.str.len() > 20]  # meaningful text only

# TF-IDF for topic discovery
tfidf = TfidfVectorizer(
    max_features=1000,
    stop_words='english',
    max_df=0.8,
    min_df=5,
    ngram_range=(1, 2)
)
tfidf_matrix = tfidf.fit_transform(text_data)

# LDA topic modeling
n_topics = 6
lda = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42,
    max_iter=20,
    learning_method='online'
)
lda_output = lda.fit_transform(tfidf_matrix)

# Extract top words per topic
feature_names = tfidf.get_feature_names_out()
topic_labels = []
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-8:-1]]
    label = f"Topic {topic_idx+1}: {', '.join(top_words[:4])}"
    topic_labels.append(label)
    print(f"  {label}")
    print(f"    All keywords: {', '.join(top_words)}")

log_insight("NLP", f"Discovered {n_topics} latent topics in {len(text_data)} posts using LDA")

# Assign dominant topic to each post with text
text_indices = text_data.index
dominant_topics = lda_output.argmax(axis=1)

# Topic distribution visualization
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('NLP Topic Modeling — Hidden Discussion Themes', fontweight='bold', fontsize=15, y=1.02)

# Topic distribution
topic_counts = pd.Series(dominant_topics).value_counts().sort_index()
bars = axes[0].barh(
    [f"Topic {i+1}" for i in topic_counts.index],
    topic_counts.values,
    color=PALETTE[:n_topics],
    edgecolor='#30363d'
)
axes[0].set_xlabel('Number of Posts')
axes[0].set_title('Topic Distribution')
for bar, count in zip(bars, topic_counts.values):
    axes[0].text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
                 f'{count}', va='center', fontsize=10, color='#c9d1d9')

# Topic word clouds (as horizontal bar charts of word importance)
top_topic = topic_counts.idxmax()
top_words_idx = lda.components_[top_topic].argsort()[:-11:-1]
top_words_vals = lda.components_[top_topic][top_words_idx]
top_words_names = [feature_names[i] for i in top_words_idx]
axes[1].barh(top_words_names[::-1], top_words_vals[::-1], color=PALETTE[top_topic], edgecolor='#30363d')
axes[1].set_title(f'Top Words — Topic {top_topic+1} (Dominant)')
axes[1].set_xlabel('Importance Score')

# Topic coherence per platform
posts_with_topics = posts.loc[text_indices].copy()
posts_with_topics['dominant_topic'] = dominant_topics
platform_topic = posts_with_topics.groupby(['platform', 'dominant_topic']).size().unstack(fill_value=0)
platform_topic_pct = platform_topic.div(platform_topic.sum(axis=1), axis=0) * 100
platform_topic_pct.plot(kind='bar', stacked=True, ax=axes[2], color=PALETTE[:n_topics], edgecolor='#30363d')
axes[2].set_title('Topic Mix by Platform')
axes[2].set_ylabel('% of Posts')
axes[2].set_xlabel('')
axes[2].legend([f'T{i+1}' for i in range(n_topics)], loc='upper right', fontsize=8)
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '14_topic_modeling.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] 14_topic_modeling.png")


# ═══════════════════════════════════════════════════════════════
# 2. USER NETWORK ANALYSIS
#    Build interaction graph, find communities and influencers
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("2. USER NETWORK ANALYSIS")
print(f"{'='*60}")

# Build a user similarity network based on shared hashtags and platforms
# Users who post with similar hashtags at similar times are "connected"

def extract_hashtags(text):
    if pd.isna(text):
        return []
    return re.findall(r'#(\w+)', str(text))

posts['hashtags'] = posts['text_content'].apply(extract_hashtags)

# Build user hashtag profiles
user_hashtags = {}
for _, row in posts.iterrows():
    uid = row['user_id']
    if uid not in user_hashtags:
        user_hashtags[uid] = Counter()
    for tag in row['hashtags']:
        user_hashtags[uid][tag] += 1

# Build network: connect users who share 3+ hashtags
G = nx.Graph()
user_ids = list(user_hashtags.keys())

# For performance, use a smarter approach: invert the hashtag->users mapping
hashtag_users = {}
for uid, tags in user_hashtags.items():
    for tag in tags:
        if tag not in hashtag_users:
            hashtag_users[tag] = set()
        hashtag_users[tag].add(uid)

# Count shared hashtags between user pairs
edge_weights = Counter()
for tag, tag_users in hashtag_users.items():
    tag_users_list = list(tag_users)
    for i in range(len(tag_users_list)):
        for j in range(i+1, min(i+50, len(tag_users_list))):  # limit for perf
            pair = tuple(sorted([tag_users_list[i], tag_users_list[j]]))
            edge_weights[pair] += 1

# Add edges for pairs sharing 3+ hashtags
for (u1, u2), weight in edge_weights.items():
    if weight >= 3:
        G.add_edge(u1, u2, weight=weight)

print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Calculate network metrics
degree_centrality = nx.degree_centrality(G)
if G.number_of_nodes() > 0:
    betweenness = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
    
    # Find top influencers
    top_by_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    top_by_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print(f"\n  Top Influencers (by degree centrality):")
    for uid, score in top_by_degree[:5]:
        post_count = len(posts[posts['user_id'] == uid])
        print(f"    {uid}: centrality={score:.4f}, posts={post_count}")
    
    log_insight("NETWORK", f"Built user interaction graph: {G.number_of_nodes()} connected users, {G.number_of_edges()} connections")
    log_insight("NETWORK", f"Top influencer: {top_by_degree[0][0]} (centrality: {top_by_degree[0][1]:.4f})")
    
    # Find communities using greedy modularity
    communities = list(nx.community.greedy_modularity_communities(G))
    log_insight("NETWORK", f"Detected {len(communities)} community clusters")
    print(f"\n  Communities detected: {len(communities)}")
    for i, comm in enumerate(communities[:5]):
        print(f"    Community {i+1}: {len(comm)} users")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('User Network Analysis — Communities & Influencers', fontweight='bold', fontsize=15, y=1.02)

if G.number_of_nodes() > 0:
    # Degree distribution
    degrees = [d for _, d in G.degree()]
    axes[0].hist(degrees, bins=30, color=PALETTE[0], edgecolor='#30363d', alpha=0.8)
    axes[0].set_title('Degree Distribution')
    axes[0].set_xlabel('Number of Connections')
    axes[0].set_ylabel('Number of Users')
    axes[0].axvline(np.mean(degrees), color=PALETTE[2], linestyle='--', label=f'Mean: {np.mean(degrees):.1f}')
    axes[0].legend()
    
    # Top influencers bar chart
    top_names = [x[0][:12] for x in top_by_degree[:10]]
    top_scores = [x[1] for x in top_by_degree[:10]]
    axes[1].barh(top_names[::-1], top_scores[::-1], color=PALETTE[1], edgecolor='#30363d')
    axes[1].set_title('Top 10 Influencers (Degree Centrality)')
    axes[1].set_xlabel('Centrality Score')
    
    # Community size distribution
    comm_sizes = [len(c) for c in communities]
    axes[2].bar(range(min(15, len(comm_sizes))), sorted(comm_sizes, reverse=True)[:15],
                color=PALETTE[3], edgecolor='#30363d')
    axes[2].set_title(f'Community Sizes ({len(communities)} clusters)')
    axes[2].set_xlabel('Community Rank')
    axes[2].set_ylabel('Users in Community')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '15_network_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] 15_network_analysis.png")


# ═══════════════════════════════════════════════════════════════
# 3. TIME SERIES DECOMPOSITION
#    Trend + Seasonality + Residual
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("3. TIME SERIES DECOMPOSITION")
print(f"{'='*60}")

# Daily posting volume
daily_posts = posts.set_index('timestamp').resample('D').size()
daily_posts = daily_posts[daily_posts > 0]

# Manual decomposition (avoiding statsmodels dependency)
# Rolling average for trend (14-day window)
window = 14
trend = daily_posts.rolling(window=window, center=True).mean()

# Detrended series
detrended = daily_posts - trend

# Seasonal component: average for each day of week
seasonal_pattern = detrended.groupby(detrended.index.dayofweek).mean()
seasonal = detrended.index.dayofweek.map(lambda x: seasonal_pattern.get(x, 0))
seasonal.index = detrended.index

# Residual
residual = detrended - seasonal

log_insight("TIMESERIES", f"Analyzed {len(daily_posts)} days of posting data")
log_insight("TIMESERIES", f"Trend shows {'growth' if trend.dropna().iloc[-1] > trend.dropna().iloc[0] else 'decline'} over the period")
log_insight("TIMESERIES", f"Day-of-week seasonality range: {seasonal_pattern.min():.1f} to {seasonal_pattern.max():.1f} posts")

# Also compute per-platform trends
platform_daily = posts.dropna(subset=['platform']).groupby(
    [pd.Grouper(key='timestamp', freq='W'), 'platform']
).size().unstack(fill_value=0)

fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

# Original series
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(daily_posts.index, daily_posts.values, color=PALETTE[0], alpha=0.5, linewidth=0.8)
ax1.plot(trend.index, trend.values, color=PALETTE[2], linewidth=2, label=f'{window}-day Moving Average')
ax1.fill_between(daily_posts.index, daily_posts.values, alpha=0.15, color=PALETTE[0])
ax1.set_title('Original Time Series with Trend', fontsize=13)
ax1.set_ylabel('Daily Posts')
ax1.legend()

# Seasonal component
ax2 = fig.add_subplot(gs[1, 0])
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
ax2.bar(days, [seasonal_pattern.get(i, 0) for i in range(7)],
        color=PALETTE[1], edgecolor='#30363d')
ax2.set_title('Day-of-Week Seasonality', fontsize=13)
ax2.set_ylabel('Deviation from Trend')
ax2.axhline(0, color='#c9d1d9', linestyle='--', alpha=0.3)

# Residual
ax3 = fig.add_subplot(gs[1, 1])
residual_clean = residual.dropna()
ax3.scatter(residual_clean.index, residual_clean.values,
            alpha=0.3, s=8, color=PALETTE[3])
ax3.axhline(0, color=PALETTE[2], linestyle='--', alpha=0.5)
ax3.set_title('Residual (Noise)', fontsize=13)
ax3.set_ylabel('Residual Value')
std_res = residual_clean.std()
log_insight("TIMESERIES", f"Residual std: {std_res:.1f} posts — {'high noise' if std_res > 5 else 'stable signal'}")

# Per-platform weekly trends
ax4 = fig.add_subplot(gs[2, :])
for i, platform in enumerate(platform_daily.columns):
    ax4.plot(platform_daily.index, platform_daily[platform],
             color=PALETTE[i], linewidth=1.5, alpha=0.8, label=platform)
ax4.set_title('Weekly Post Volume by Platform', fontsize=13)
ax4.set_ylabel('Posts per Week')
ax4.legend(loc='upper right')

fig.suptitle('Time Series Decomposition — Trend, Seasonality & Noise', fontweight='bold', fontsize=15, y=1.01)
plt.savefig(os.path.join(PLOTS_DIR, '16_timeseries_decomposition.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] 16_timeseries_decomposition.png")


# ═══════════════════════════════════════════════════════════════
# 4. ENGAGEMENT PREDICTION MODEL
#    Random Forest + Feature Importance
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("4. ENGAGEMENT PREDICTION MODEL")
print(f"{'='*60}")

# Feature engineering
model_data = posts.dropna(subset=['likes', 'platform', 'text_content']).copy()
model_data['text_length'] = model_data['text_content'].str.len()
model_data['word_count'] = model_data['text_content'].str.split().str.len()
model_data['hashtag_count'] = model_data['text_content'].apply(lambda x: len(re.findall(r'#\w+', str(x))))
model_data['has_question'] = model_data['text_content'].str.contains(r'\?', na=False).astype(int)
model_data['hour'] = model_data['timestamp'].dt.hour
model_data['day_of_week'] = model_data['timestamp'].dt.dayofweek
model_data['is_weekend'] = (model_data['day_of_week'] >= 5).astype(int)
model_data['month'] = model_data['timestamp'].dt.month

# Sentiment encoding
sentiment_map = {'Positive': 2, 'Neutral': 1, 'Negative': 0, 'Unknown': 1}
# Detect sentiment from text for feature
def quick_sentiment(text):
    if pd.isna(text):
        return 1
    text = text.lower()
    pos_words = ['love', 'great', 'amazing', 'best', 'excellent', 'fantastic', 'awesome', 'perfect', 'recommend']
    neg_words = ['hate', 'worst', 'terrible', 'awful', 'bad', 'horrible', 'disappointed', 'poor', 'waste']
    pos = sum(1 for w in pos_words if w in text)
    neg = sum(1 for w in neg_words if w in text)
    if pos > neg:
        return 2
    elif neg > pos:
        return 0
    return 1

model_data['sentiment_score'] = model_data['text_content'].apply(quick_sentiment)

# Platform encoding
le = LabelEncoder()
model_data['platform_encoded'] = le.fit_transform(model_data['platform'])

# Merge user features
user_features = users[['user_id', 'follower_count']].copy()
model_data = model_data.merge(user_features, on='user_id', how='left')

feature_cols = ['text_length', 'word_count', 'hashtag_count', 'has_question',
                'hour', 'day_of_week', 'is_weekend', 'month',
                'sentiment_score', 'platform_encoded', 'follower_count',
                'shares', 'comments']

X = model_data[feature_cols].fillna(0)
y = model_data['likes']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n  Random Forest Results:")
print(f"    MAE: {mae:.1f} likes")
print(f"    R-squared: {r2:.4f}")
print(f"    Train samples: {len(X_train)}, Test samples: {len(X_test)}")

log_insight("ML_MODEL", f"Random Forest engagement predictor: MAE={mae:.1f} likes, R2={r2:.4f}")

# Feature importance
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(f"\n  Feature Importance Ranking:")
for feat, imp in importances.items():
    print(f"    {feat:25s} {imp:.4f}")

top_feature = importances.index[0]
log_insight("ML_MODEL", f"Top predictor of engagement: '{top_feature}' (importance: {importances.iloc[0]:.4f})")

# Gradient Boosting for comparison
gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_mae = mean_absolute_error(y_test, gb_pred)
gb_r2 = r2_score(y_test, gb_pred)
print(f"\n  Gradient Boosting Results:")
print(f"    MAE: {gb_mae:.1f} likes")
print(f"    R-squared: {gb_r2:.4f}")
log_insight("ML_MODEL", f"Gradient Boosting: MAE={gb_mae:.1f} likes, R2={gb_r2:.4f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Engagement Prediction Model — What Drives Likes?', fontweight='bold', fontsize=15, y=1.01)

# Feature importance
axes[0, 0].barh(importances.index[::-1], importances.values[::-1],
                color=PALETTE[0], edgecolor='#30363d')
axes[0, 0].set_title('Feature Importance (Random Forest)')
axes[0, 0].set_xlabel('Importance Score')

# Actual vs Predicted scatter
sample_idx = np.random.choice(len(y_test), min(2000, len(y_test)), replace=False)
axes[0, 1].scatter(y_test.iloc[sample_idx], y_pred[sample_idx],
                   alpha=0.2, s=10, color=PALETTE[1])
axes[0, 1].plot([0, 5000], [0, 5000], color=PALETTE[2], linestyle='--', linewidth=2, label='Perfect Prediction')
axes[0, 1].set_title(f'Actual vs Predicted Likes (R2={r2:.3f})')
axes[0, 1].set_xlabel('Actual Likes')
axes[0, 1].set_ylabel('Predicted Likes')
axes[0, 1].legend()

# Residual distribution
residuals = y_test.values - y_pred
axes[1, 0].hist(residuals, bins=50, color=PALETTE[3], edgecolor='#30363d', alpha=0.8)
axes[1, 0].axvline(0, color=PALETTE[2], linestyle='--', linewidth=2)
axes[1, 0].set_title('Prediction Error Distribution')
axes[1, 0].set_xlabel('Error (Actual - Predicted)')
axes[1, 0].set_ylabel('Frequency')

# Model comparison
models = ['Random Forest', 'Gradient Boosting']
maes = [mae, gb_mae]
r2s = [r2, gb_r2]
x_pos = np.arange(len(models))
width = 0.35
bars1 = axes[1, 1].bar(x_pos - width/2, maes, width, label='MAE', color=PALETTE[0], edgecolor='#30363d')
ax_twin = axes[1, 1].twinx()
bars2 = ax_twin.bar(x_pos + width/2, r2s, width, label='R-squared', color=PALETTE[1], edgecolor='#30363d')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(models)
axes[1, 1].set_ylabel('MAE (likes)')
ax_twin.set_ylabel('R-squared')
axes[1, 1].set_title('Model Comparison')
axes[1, 1].legend(loc='upper left')
ax_twin.legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '17_engagement_prediction.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] 17_engagement_prediction.png")


# ═══════════════════════════════════════════════════════════════
# 5. USER SEGMENTATION (K-Means Clustering)
#    Discover user archetypes
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("5. USER SEGMENTATION (K-Means)")
print(f"{'='*60}")

# Build user feature matrix
user_stats = posts.groupby('user_id').agg(
    total_posts=('post_id', 'count'),
    avg_likes=('likes', 'mean'),
    avg_shares=('shares', 'mean'),
    avg_comments=('comments', 'mean'),
    platform_count=('platform', 'nunique'),
    avg_text_length=('text_content', lambda x: x.dropna().str.len().mean()),
).reset_index()

user_stats = user_stats.merge(users[['user_id', 'follower_count']], on='user_id', how='left')
user_stats = user_stats.fillna(0)

# Normalize for clustering
from sklearn.preprocessing import StandardScaler
cluster_features = ['total_posts', 'avg_likes', 'avg_shares', 'avg_comments',
                    'platform_count', 'follower_count', 'avg_text_length']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(user_stats[cluster_features])

# Elbow method to find optimal K
inertias = []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# Use K=4 (reasonable for user segmentation)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
user_stats['cluster'] = kmeans.fit_predict(X_scaled)

# Profile each cluster
print(f"\n  User Segments ({optimal_k} clusters):")
cluster_profiles = user_stats.groupby('cluster')[cluster_features].mean()
for cluster_id in range(optimal_k):
    profile = cluster_profiles.loc[cluster_id]
    # Auto-name the cluster
    if profile['follower_count'] > cluster_profiles['follower_count'].median() and profile['avg_likes'] > cluster_profiles['avg_likes'].median():
        name = "Power Users"
    elif profile['total_posts'] > cluster_profiles['total_posts'].median():
        name = "Active Contributors"
    elif profile['avg_text_length'] > cluster_profiles['avg_text_length'].median():
        name = "Long-form Writers"
    else:
        name = "Casual Browsers"
    
    print(f"\n    Cluster {cluster_id+1} — '{name}' ({len(user_stats[user_stats['cluster']==cluster_id])} users)")
    print(f"      Avg posts: {profile['total_posts']:.1f}, Avg likes: {profile['avg_likes']:.0f}")
    print(f"      Avg followers: {profile['follower_count']:.0f}, Platforms: {profile['platform_count']:.1f}")

log_insight("SEGMENTATION", f"Identified {optimal_k} distinct user segments via K-Means clustering")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('User Segmentation — K-Means Clustering', fontweight='bold', fontsize=15, y=1.01)

# Elbow plot
axes[0, 0].plot(list(K_range), inertias, 'o-', color=PALETTE[0], linewidth=2, markersize=8)
axes[0, 0].axvline(optimal_k, color=PALETTE[2], linestyle='--', label=f'Chosen K={optimal_k}')
axes[0, 0].set_title('Elbow Method')
axes[0, 0].set_xlabel('Number of Clusters (K)')
axes[0, 0].set_ylabel('Inertia')
axes[0, 0].legend()

# Scatter: followers vs avg likes, colored by cluster
for c in range(optimal_k):
    mask = user_stats['cluster'] == c
    axes[0, 1].scatter(user_stats.loc[mask, 'follower_count'],
                       user_stats.loc[mask, 'avg_likes'],
                       alpha=0.4, s=15, color=PALETTE[c], label=f'Cluster {c+1}')
axes[0, 1].set_title('User Segments: Followers vs Avg Likes')
axes[0, 1].set_xlabel('Follower Count')
axes[0, 1].set_ylabel('Average Likes')
axes[0, 1].legend()

# Cluster size
cluster_sizes = user_stats['cluster'].value_counts().sort_index()
axes[1, 0].bar([f'Cluster {i+1}' for i in cluster_sizes.index], cluster_sizes.values,
               color=PALETTE[:optimal_k], edgecolor='#30363d')
axes[1, 0].set_title('Cluster Sizes')
axes[1, 0].set_ylabel('Number of Users')

# Radar chart — cluster profiles (simplified as grouped bar)
cluster_means = user_stats.groupby('cluster')[['total_posts', 'avg_likes', 'follower_count']].mean()
cluster_means_norm = cluster_means.div(cluster_means.max())
x_r = np.arange(len(cluster_means_norm.columns))
width = 0.2
for c in range(optimal_k):
    axes[1, 1].bar(x_r + c*width, cluster_means_norm.iloc[c].values,
                   width, color=PALETTE[c], label=f'Cluster {c+1}', edgecolor='#30363d')
axes[1, 1].set_xticks(x_r + width*(optimal_k-1)/2)
axes[1, 1].set_xticklabels(['Posts', 'Avg Likes', 'Followers'])
axes[1, 1].set_title('Normalized Cluster Profiles')
axes[1, 1].set_ylabel('Normalized Value')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '18_user_segmentation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] 18_user_segmentation.png")


# ═══════════════════════════════════════════════════════════════
# 6. VIRAL POST ANALYSIS
#    What makes a post go viral? Anatomy of top performers
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("6. VIRAL POST ANALYSIS")
print(f"{'='*60}")

posts_with_engagement = posts.dropna(subset=['likes']).copy()
posts_with_engagement['total_engagement'] = (
    posts_with_engagement['likes'] +
    posts_with_engagement['shares'] +
    posts_with_engagement['comments']
)

# Define viral threshold (top 5%)
viral_threshold = posts_with_engagement['total_engagement'].quantile(0.95)
posts_with_engagement['is_viral'] = posts_with_engagement['total_engagement'] >= viral_threshold

viral_posts = posts_with_engagement[posts_with_engagement['is_viral']]
normal_posts = posts_with_engagement[~posts_with_engagement['is_viral']]

print(f"  Viral threshold (top 5%): {viral_threshold:.0f} total engagement")
print(f"  Viral posts: {len(viral_posts)}, Normal posts: {len(normal_posts)}")

# Compare viral vs normal
viral_text = viral_posts['text_content'].dropna()
normal_text = normal_posts['text_content'].dropna()

viral_avg_len = viral_text.str.len().mean()
normal_avg_len = normal_text.str.len().mean()
viral_hashtags = viral_text.apply(lambda x: len(re.findall(r'#\w+', str(x)))).mean()
normal_hashtags = normal_text.apply(lambda x: len(re.findall(r'#\w+', str(x)))).mean()
viral_questions = viral_text.str.contains(r'\?').mean() * 100
normal_questions = normal_text.str.contains(r'\?').mean() * 100

print(f"\n  Viral vs Normal Comparison:")
print(f"    Avg text length:  {viral_avg_len:.0f} vs {normal_avg_len:.0f}")
print(f"    Avg hashtags:     {viral_hashtags:.2f} vs {normal_hashtags:.2f}")
print(f"    Questions:        {viral_questions:.1f}% vs {normal_questions:.1f}%")

log_insight("VIRAL", f"Viral threshold: top 5% = {viral_threshold:.0f}+ total engagement")
log_insight("VIRAL", f"Viral posts avg text length: {viral_avg_len:.0f} chars vs normal: {normal_avg_len:.0f}")

# Platform breakdown of viral posts
viral_platform = viral_posts['platform'].value_counts()

# Hour breakdown
viral_hours = viral_posts['timestamp'].dt.hour.value_counts().sort_index()

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Viral Post Anatomy — What Makes Content Explode?', fontweight='bold', fontsize=15, y=1.01)

# Engagement distribution with viral threshold
axes[0, 0].hist(posts_with_engagement['total_engagement'], bins=50,
                color=PALETTE[0], edgecolor='#30363d', alpha=0.7, label='All Posts')
axes[0, 0].axvline(viral_threshold, color=PALETTE[2], linestyle='--', linewidth=2,
                   label=f'Viral Threshold ({viral_threshold:.0f})')
axes[0, 0].set_title('Engagement Distribution')
axes[0, 0].set_xlabel('Total Engagement (Likes + Shares + Comments)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()

# Viral vs Normal comparison
comparison_metrics = ['Text Length', 'Hashtags', 'Questions %']
viral_vals = [viral_avg_len, viral_hashtags * 100, viral_questions]
normal_vals = [normal_avg_len, normal_hashtags * 100, normal_questions]
x = np.arange(len(comparison_metrics))
width = 0.35
axes[0, 1].bar(x - width/2, viral_vals, width, label='Viral (Top 5%)', color=PALETTE[2], edgecolor='#30363d')
axes[0, 1].bar(x + width/2, normal_vals, width, label='Normal', color=PALETTE[0], edgecolor='#30363d')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(comparison_metrics)
axes[0, 1].set_title('Viral vs Normal Posts')
axes[0, 1].legend()

# Viral posts by platform
if not viral_platform.empty:
    axes[1, 0].pie(viral_platform.values, labels=viral_platform.index,
                   colors=PALETTE[:len(viral_platform)], autopct='%1.1f%%',
                   textprops={'color': '#c9d1d9'})
    axes[1, 0].set_title('Viral Posts by Platform')

# Viral post timing
if not viral_hours.empty:
    axes[1, 1].bar(viral_hours.index, viral_hours.values,
                   color=PALETTE[1], edgecolor='#30363d')
    axes[1, 1].set_title('When Do Viral Posts Happen?')
    axes[1, 1].set_xlabel('Hour of Day')
    axes[1, 1].set_ylabel('Number of Viral Posts')
    axes[1, 1].set_xticks(range(0, 24, 2))

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '19_viral_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] 19_viral_analysis.png")


# ═══════════════════════════════════════════════════════════════
# 7. CROSS-PLATFORM USER BEHAVIOR
#    Do multi-platform users behave differently?
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("7. CROSS-PLATFORM USER BEHAVIOR")
print(f"{'='*60}")

# Users by number of platforms
user_platform_count = posts.dropna(subset=['platform']).groupby('user_id')['platform'].nunique()
user_multi = user_platform_count.value_counts().sort_index()

print(f"  Users by platform count:")
for count, n_users in user_multi.items():
    print(f"    {count} platform(s): {n_users} users")

# Multi-platform vs single-platform engagement
single_users = user_platform_count[user_platform_count == 1].index
multi_users = user_platform_count[user_platform_count >= 3].index

single_engagement = posts[posts['user_id'].isin(single_users)]['likes'].mean()
multi_engagement = posts[posts['user_id'].isin(multi_users)]['likes'].mean()

log_insight("CROSSPLATFORM", f"Multi-platform users (3+) avg likes: {multi_engagement:.0f} vs single-platform: {single_engagement:.0f}")

# Platform migration: which platforms do users typically combine?
user_platforms = posts.dropna(subset=['platform']).groupby('user_id')['platform'].apply(set)
platform_combos = Counter()
for platforms in user_platforms:
    if len(platforms) >= 2:
        for p in platforms:
            for q in platforms:
                if p < q:
                    platform_combos[(p, q)] += 1

print(f"\n  Top platform combinations:")
for (p1, p2), count in platform_combos.most_common(5):
    print(f"    {p1} + {p2}: {count} users")

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('Cross-Platform User Behavior', fontweight='bold', fontsize=15, y=1.02)

# Users by platform count
axes[0].bar(user_multi.index.astype(str), user_multi.values,
            color=PALETTE[0], edgecolor='#30363d')
axes[0].set_title('Users by Number of Platforms')
axes[0].set_xlabel('Number of Platforms Used')
axes[0].set_ylabel('Number of Users')

# Engagement comparison
axes[1].bar(['Single Platform', 'Multi-Platform (3+)'],
            [single_engagement, multi_engagement],
            color=[PALETTE[2], PALETTE[1]], edgecolor='#30363d')
axes[1].set_title('Avg Likes: Single vs Multi-Platform Users')
axes[1].set_ylabel('Average Likes')

# Platform co-occurrence heatmap
all_platforms = ['Facebook', 'Instagram', 'Reddit', 'Twitter', 'YouTube']
cooccurrence = pd.DataFrame(0, index=all_platforms, columns=all_platforms)
for (p1, p2), count in platform_combos.items():
    cooccurrence.loc[p1, p2] = count
    cooccurrence.loc[p2, p1] = count
sns.heatmap(cooccurrence, annot=True, fmt='d', cmap='YlOrRd', ax=axes[2],
            linewidths=0.5, linecolor='#30363d')
axes[2].set_title('Platform Co-occurrence (User Overlap)')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '20_crossplatform_behavior.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] 20_crossplatform_behavior.png")


# ═══════════════════════════════════════════════════════════════
# SAVE ADVANCED INSIGHTS
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("ADVANCED ANALYTICS COMPLETE")
print(f"{'='*60}")

with open(os.path.join(DATA_DIR, 'advanced_insights.txt'), 'w', encoding='utf-8') as f:
    f.write("SOCIAL ENGINE RECOVERY — ADVANCED ANALYTICS INSIGHTS\n")
    f.write("=" * 60 + "\n\n")
    for insight in advanced_insights:
        f.write(insight + "\n")

print(f"\nAdvanced insights saved to: advanced_insights.txt")
print(f"\nNew plots generated:")
print(f"  14_topic_modeling.png")
print(f"  15_network_analysis.png")
print(f"  16_timeseries_decomposition.png")
print(f"  17_engagement_prediction.png")
print(f"  18_user_segmentation.png")
print(f"  19_viral_analysis.png")
print(f"  20_crossplatform_behavior.png")
