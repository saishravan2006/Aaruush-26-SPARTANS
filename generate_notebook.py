import json
import os

def create_cell(cell_type, source):
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip()

DATA_DIR = r"C:\Users\saish\.gemini\antigravity-ide\scratch\social-engine-recovery"
EXTRACT_PATH = r"C:\Users\saish\.gemini\antigravity-ide\brain\f29b1de3-9ae7-424c-9060-f21c0039f01f\scratch\extract_data.py"

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# ── Title ──
notebook["cells"].append(create_cell("markdown", """# Social Engine Recovery — Complete Analysis
### Data Vortex :: AARUUSH'26 — Round 1 Phase 1

> **Mission**: The Social Engine suffered a critical failure, corrupting its data intake and analytical capabilities. This notebook documents the complete recovery process — from extracting the hidden dataset, through cleaning the corrupted pipeline artifacts, to advanced machine learning analysis.

---

## Table of Contents
1. **Data Extraction** — Recovering the dataset from the crashed node_07 archive
2. **Data Cleaning Pipeline** — Systematic decontamination of 8 corruption patterns
3. **Exploratory Data Analysis** — 13 visualization panels covering platform, temporal, engagement, brand, sentiment, and geographic analysis
4. **Advanced Analytics** — NLP topic modeling, network analysis, time series decomposition, ML engagement prediction, user segmentation, viral post anatomy, and cross-platform behavior"""))

# ── Part 1: Extraction ──
notebook["cells"].append(create_cell("markdown", """---
## 1. Data Extraction

The corrupted dataset was not provided directly — it was hidden inside the crashed Social Engine dashboard at `https://datavortex-social-engine.vercel.app/`.

**Discovery Process:**
1. Inspected the React application's JS bundle (`index-B2FT5USN.js`)
2. Used the built-in recovery terminal: ran `logs` to identify `node_07` as the last surviving node
3. Ran `connect node_07` to access the archive containing the CSV data
4. Extracted the data from template literals embedded in the JavaScript bundle

The following script automates the extraction:"""))
notebook["cells"].append(create_cell("code", read_file(EXTRACT_PATH)))

# ── Part 2: Cleaning ──
notebook["cells"].append(create_cell("markdown", """---
## 2. Data Cleaning Pipeline

The system failure introduced **8 distinct corruption patterns** into the dataset:

| # | Corruption | Affected Rows | Resolution |
|---|-----------|--------------|------------|
| 1 | Mixed timestamp formats (ISO 8601, DD-MM-YYYY, Unix epoch) | 12,000 | Normalized to datetime64 |
| 2 | Exact duplicate rows (re-ingestion artifacts) | 360 | Removed |
| 3 | Missing values (platform, text, likes) | ~1,800 | Preserved as NaN (unrecoverable) |
| 4 | Negative like counts | 509 | Set to NaN (impossible values) |
| 5 | HTML tags in text (`<br>`, `<div>`) | 646 | Stripped |
| 6 | UTF-8 mojibake encoding (`Ã©`) | 306 | Cleaned |
| 7 | Embedded whitespace/newlines | 329 | Normalized |
| 8 | Missing platform identifiers | 1,784 | Validated against known set |

**Result**: 12,360 raw rows → 12,000 clean rows with full data provenance."""))
notebook["cells"].append(create_cell("code", read_file(os.path.join(DATA_DIR, "data_cleaning.py"))))

# ── Part 3: Basic EDA ──
notebook["cells"].append(create_cell("markdown", """---
## 3. Exploratory Data Analysis (EDA)

Comprehensive analysis across 13 dimensions:
- **Platform Distribution** — Post volume and share across YouTube, Facebook, Twitter, Reddit, Instagram
- **Temporal Patterns** — Monthly trends, day-of-week, hourly patterns
- **Engagement Metrics** — Likes, shares, comments distributions and correlations
- **Brand & Product Mentions** — Top 10 brands and their product-level breakdowns
- **Hashtag Analysis** — 29 unique hashtags across 20,531 total mentions
- **Sentiment Analysis** — Positive (33.5%), Negative (28.3%), Neutral distribution
- **User Activity** — Post frequency, power users, follower correlations
- **Geographic & Language** — 56 cities, 10 languages represented
- **Anomaly Detection** — IQR-based outlier analysis
- **Missing Data Profiling** — Corruption impact assessment"""))
notebook["cells"].append(create_cell("code", read_file(os.path.join(DATA_DIR, "eda_analysis.py"))))

# ── Part 4: Advanced Analytics ──
notebook["cells"].append(create_cell("markdown", """---
## 4. Advanced Analytics

This section goes beyond descriptive EDA into **predictive and structural analysis**:

### 4.1 NLP Topic Modeling (LDA)
Latent Dirichlet Allocation discovers 6 hidden discussion themes using TF-IDF vectorization. Reveals what users are *actually* talking about beyond simple keyword counts.

### 4.2 User Network Analysis
Builds a graph of 1,488 connected users with 132K+ edges based on shared hashtag usage. Identifies influencer nodes via degree centrality and detects 3 community clusters using greedy modularity optimization.

### 4.3 Time Series Decomposition
Decomposes daily posting volume into **Trend** (14-day moving average), **Seasonality** (day-of-week patterns), and **Residual** (noise) components. Shows overall growth trend with high noise (std=5.8 posts).

### 4.4 Engagement Prediction (ML)
Random Forest and Gradient Boosting regressors predict likes from 13 engineered features. Feature importance reveals that **follower count** (16.9%) and **shares** (16.7%) are the strongest engagement predictors.

### 4.5 User Segmentation (K-Means)
Clusters 1,500 users into 4 distinct archetypes: Power Users, Active Contributors, Long-form Writers, and high-follower Active Contributors. Uses elbow method for optimal K selection.

### 4.6 Viral Post Anatomy
Dissects the top 5% of posts (6,560+ total engagement) to find what makes content explode. Compares text length, hashtag usage, questioning patterns, platform distribution, and timing.

### 4.7 Cross-Platform User Behavior
Analyzes multi-platform usage patterns. 88.7% of users are active on 3+ platforms. Facebook+Reddit is the most common platform combination (873 users)."""))
notebook["cells"].append(create_cell("code", read_file(os.path.join(DATA_DIR, "advanced_analysis.py"))))

# ── Conclusion ──
notebook["cells"].append(create_cell("markdown", """---
## Key Findings Summary

| Insight | Detail |
|---------|--------|
| **Dataset Recovery** | 12,000 posts + 1,500 users extracted from crashed node_07 |
| **Corruption Cleaned** | 8 distinct patterns addressed with full audit trail |
| **Platform Leader** | YouTube (17.3%), all platforms roughly equal |
| **Top Brand** | Adidas (1,070 mentions), followed by Nike |
| **Sentiment Split** | 33.5% positive, 28.3% negative |
| **Top Influencer** | user_n0ok02rt (centrality: 0.184) |
| **Community Structure** | 3 major community clusters detected |
| **Engagement Driver** | Follower count is #1 predictor (16.9% importance) |
| **User Segments** | 4 distinct archetypes via K-Means |
| **Viral Threshold** | Top 5% = 6,560+ total engagement |
| **Cross-Platform** | 88.7% of users active on 3+ platforms |

---
*Social Engine Recovery Team — Data Vortex :: AARUUSH'26*"""))

# Save
with open(os.path.join(DATA_DIR, "Social_Engine_Recovery.ipynb"), "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Social_Engine_Recovery.ipynb rebuilt with all 4 sections!")
