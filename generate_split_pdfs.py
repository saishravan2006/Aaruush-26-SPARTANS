"""
Split Phase 2 into 4 separate submission files:
1. SQL_Queries.pdf - Just the SQL queries
2. Output screenshots (JPEG) - up to 5 images
3. Logic_Explanation.pdf - Just the logic explanations
4. Phase2_Insight_Report.pdf - Insight report
"""

from fpdf import FPDF
import json
import pandas as pd
from io import StringIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def safe(text):
    replacements = {
        '\u2014': '-', '\u2013': '-', '\u2019': "'", '\u2018': "'",
        '\u201c': '"', '\u201d': '"', '\u2265': '>=', '\u2264': '<=',
        '\u2192': '->', '\u2026': '...', '\u00e3': 'a', '\u00a0': ' ',
        '\u2022': '*', '\u00e9': 'e',
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode('latin-1', 'replace').decode('latin-1')

# Load results
with open('phase2_sql_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# ═══════════════════════════════════════════
# FILE 1: SQL_Queries.pdf
# ═══════════════════════════════════════════
print("Generating SQL_Queries.pdf...")

class QueryPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 9)
        self.cell(0, 7, safe('SQL Queries - Social Engine Recovery | Team SPARTANS'), 0, new_x="LMARGIN", new_y="NEXT", align='C')
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, align='C')

pdf1 = QueryPDF('P', 'mm', 'A4')
pdf1.set_auto_page_break(auto=True, margin=20)

# Schema page
pdf1.add_page()
pdf1.set_font('Helvetica', 'B', 16)
pdf1.cell(0, 10, 'Schema Design', new_x="LMARGIN", new_y="NEXT")
pdf1.ln(3)
pdf1.set_font('Courier', '', 8)
schema_sql = """-- Database: SQLite 3
-- Schema: Normalized relational design

CREATE TABLE users (
    user_id         TEXT PRIMARY KEY,
    location        TEXT NOT NULL,
    language        TEXT NOT NULL,
    account_created TEXT NOT NULL,
    follower_count  INTEGER NOT NULL
);

CREATE TABLE posts (
    post_id      TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    platform     TEXT,
    text_content TEXT,
    timestamp    TEXT NOT NULL,
    likes        REAL,
    shares       INTEGER NOT NULL,
    comments     INTEGER NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE posts_corrupted (
    post_id TEXT PRIMARY KEY, user_id TEXT,
    platform TEXT, text_content TEXT,
    timestamp TEXT, likes TEXT,
    shares TEXT, comments TEXT
);

-- Performance Indexes
CREATE INDEX idx_posts_user_id ON posts(user_id);
CREATE INDEX idx_posts_platform ON posts(platform);
CREATE INDEX idx_users_follower_count ON users(follower_count);"""

pdf1.set_fill_color(240, 240, 240)
for line in schema_sql.split('\n'):
    pdf1.cell(0, 4, safe(line), new_x="LMARGIN", new_y="NEXT", fill=True)

# Query pages
for r in results:
    pdf1.add_page()
    pdf1.set_font('Helvetica', 'B', 13)
    pdf1.set_fill_color(44, 62, 80)
    pdf1.set_text_color(255, 255, 255)
    pdf1.cell(0, 9, safe(f"  {r['id']} - {r['title']}"), new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf1.set_text_color(0, 0, 0)
    pdf1.ln(4)
    
    pdf1.set_font('Courier', '', 8)
    pdf1.set_fill_color(240, 240, 240)
    for line in r['sql'].split('\n'):
        pdf1.cell(0, 4.5, safe(f"  {line}"), new_x="LMARGIN", new_y="NEXT", fill=True)

pdf1.output('SQL_Queries.pdf')
print("  SQL_Queries.pdf done!")


# ═══════════════════════════════════════════
# FILE 2: Output Screenshots (JPEG, up to 5)
# ═══════════════════════════════════════════
print("Generating output screenshots...")

# Group questions into 5 images
groups = [
    ('Easy_E1_to_E5', ['E1', 'E2', 'E3', 'E4', 'E5']),
    ('Medium_M1_to_M3', ['M1', 'M2', 'M3']),
    ('Medium_M4_M5', ['M4', 'M5']),
    ('Hard_H1_to_H3', ['H1', 'H2', 'H3']),
    ('Hard_H4_to_H6', ['H4', 'H5', 'H6']),
]

result_map = {r['id']: r for r in results}

for group_name, q_ids in groups:
    # Calculate total rows needed
    total_text_lines = 0
    sections = []
    for qid in q_ids:
        r = result_map[qid]
        df = pd.read_csv(StringIO(r['result_csv']))
        display_df = df.head(15)
        # Header + separator + table header + rows
        section_lines = 3 + min(len(display_df), 15) + 1
        if len(df) > 15:
            section_lines += 1
        sections.append((qid, r['title'], display_df, len(df)))
        total_text_lines += section_lines
    
    fig_height = max(6, total_text_lines * 0.28 + 1)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis('off')
    
    y_pos = 0.98
    line_height = 1.0 / (total_text_lines + 4)
    
    for qid, title, display_df, total_rows in sections:
        # Section header
        ax.text(0.01, y_pos, f"{qid} - {title}", transform=ax.transAxes,
                fontsize=11, fontweight='bold', fontfamily='monospace',
                color='#2c3e50')
        y_pos -= line_height * 1.3
        
        # Separator
        ax.plot([0.01, 0.99], [y_pos + line_height * 0.3, y_pos + line_height * 0.3],
                color='#bdc3c7', linewidth=0.5, transform=ax.transAxes)
        
        if len(display_df) == 0:
            ax.text(0.02, y_pos, "No results found (empty result set)", 
                    transform=ax.transAxes, fontsize=8, fontfamily='monospace',
                    color='#e74c3c')
            y_pos -= line_height
        else:
            # Column headers
            header_text = '  '.join(f"{col:<20}" for col in display_df.columns)
            ax.text(0.02, y_pos, header_text[:140], transform=ax.transAxes,
                    fontsize=7, fontweight='bold', fontfamily='monospace',
                    color='#2c3e50')
            y_pos -= line_height
            
            # Data rows
            for _, row in display_df.iterrows():
                row_text = '  '.join(f"{str(v):<20}" for v in row.values)
                ax.text(0.02, y_pos, row_text[:140], transform=ax.transAxes,
                        fontsize=6.5, fontfamily='monospace', color='#34495e')
                y_pos -= line_height
            
            if total_rows > 15:
                ax.text(0.02, y_pos, f"  ... {total_rows - 15} more rows",
                        transform=ax.transAxes, fontsize=7, fontfamily='monospace',
                        color='#7f8c8d', style='italic')
                y_pos -= line_height
        
        y_pos -= line_height * 0.5
    
    plt.tight_layout()
    filepath = f'output_{group_name}.jpeg'
    plt.savefig(filepath, dpi=150, bbox_inches='tight', 
                facecolor='white', format='jpeg')
    plt.close()
    print(f"  {filepath} done!")


# ═══════════════════════════════════════════
# FILE 3: Logic_Explanation.pdf
# ═══════════════════════════════════════════
print("Generating Logic_Explanation.pdf...")

class LogicPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 9)
        self.cell(0, 7, safe('Logic Explanation - Social Engine Recovery | Team SPARTANS'), 0, new_x="LMARGIN", new_y="NEXT", align='C')
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, align='C')

pdf3 = LogicPDF('P', 'mm', 'A4')
pdf3.set_auto_page_break(auto=True, margin=20)
pdf3.add_page()

pdf3.set_font('Helvetica', 'B', 18)
pdf3.cell(0, 12, 'Logic Explanation', new_x="LMARGIN", new_y="NEXT", align='C')
pdf3.set_font('Helvetica', '', 11)
pdf3.cell(0, 8, "Data Vortex :: AARUUSH'26 | Team SPARTANS", new_x="LMARGIN", new_y="NEXT", align='C')
pdf3.ln(8)

for r in results:
    if pdf3.get_y() > 240:
        pdf3.add_page()
    
    pdf3.set_font('Helvetica', 'B', 12)
    pdf3.set_fill_color(44, 62, 80)
    pdf3.set_text_color(255, 255, 255)
    pdf3.cell(0, 8, safe(f"  {r['id']} - {r['title']}"), new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf3.set_text_color(0, 0, 0)
    pdf3.ln(2)
    
    pdf3.set_font('Helvetica', '', 10)
    pdf3.multi_cell(0, 6, text=safe(r['logic']), new_x="LMARGIN", new_y="NEXT")
    
    # Add SQL techniques used
    pdf3.set_font('Helvetica', 'I', 9)
    techniques = []
    sql = r['sql'].upper()
    if 'WITH' in sql: techniques.append('CTE')
    if 'OVER' in sql: techniques.append('Window Function')
    if 'CASE' in sql: techniques.append('CASE Expression')
    if 'JOIN' in sql: techniques.append('JOIN')
    if 'COALESCE' in sql: techniques.append('COALESCE')
    if 'GROUP BY' in sql: techniques.append('GROUP BY')
    if 'HAVING' in sql: techniques.append('HAVING')
    if 'OFFSET' in sql: techniques.append('Subquery/OFFSET')
    if techniques:
        pdf3.cell(0, 5, safe(f"  SQL Techniques: {', '.join(techniques)}"), new_x="LMARGIN", new_y="NEXT")
    pdf3.ln(4)

pdf3.output('Logic_Explanation.pdf')
print("  Logic_Explanation.pdf done!")


# ═══════════════════════════════════════════
# FILE 4: Phase2_Insight_Report.pdf
# ═══════════════════════════════════════════
print("Generating Phase2_Insight_Report.pdf...")

class InsightPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 9)
        self.cell(0, 7, safe('Phase 2 Insight Report - Social Engine Recovery | Team SPARTANS'), 0, new_x="LMARGIN", new_y="NEXT", align='C')
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, align='C')

pdf4 = InsightPDF('P', 'mm', 'A4')
pdf4.set_auto_page_break(auto=True, margin=20)
pdf4.add_page()

pdf4.ln(15)
pdf4.set_font('Helvetica', 'B', 22)
pdf4.cell(0, 12, 'Phase 2 Insight Report', new_x="LMARGIN", new_y="NEXT", align='C')
pdf4.set_font('Helvetica', '', 14)
pdf4.cell(0, 10, 'Social Engine Recovery - SQL Analytics', new_x="LMARGIN", new_y="NEXT", align='C')
pdf4.set_font('Helvetica', 'B', 12)
pdf4.cell(0, 8, "Data Vortex :: AARUUSH'26 | Team SPARTANS", new_x="LMARGIN", new_y="NEXT", align='C')
pdf4.ln(15)

# Executive Summary
pdf4.set_font('Helvetica', 'B', 14)
pdf4.cell(0, 8, 'Executive Summary', new_x="LMARGIN", new_y="NEXT")
pdf4.ln(2)
pdf4.set_font('Helvetica', '', 10)
pdf4.multi_cell(0, 6, text=safe(
    'After restoring the Social Engine dataset in Phase 1, we loaded the cleaned data into a '
    'normalized SQLite database and executed 16 analytical SQL queries across Easy, Medium, and Hard '
    'difficulty tiers. Our analysis uncovered critical insights about platform engagement patterns, '
    'user behaviour anomalies, and data corruption profiles. Below are our key findings.'
), new_x="LMARGIN", new_y="NEXT")
pdf4.ln(5)

# Key Findings
insights = [
    ("1. Platform Landscape is Balanced", 
     "All 5 platforms (Facebook, YouTube, Twitter, Reddit, Instagram) have nearly equal post volumes, within 4% of each other. Facebook leads by a slim margin (2,074 posts). This balanced distribution suggests the Social Engine had equal platform integration before the crash."),
    
    ("2. YouTube Drives Highest Per-Post Engagement",
     "Despite Facebook having the most posts, YouTube generates the highest average total engagement at 4,044.54 per post. This confirms that video-based content consistently outperforms text-only platforms in driving user interaction."),
    
    ("3. The Follower Count Paradox",
     "Our M2 analysis reveals a counter-intuitive finding: users with fewer than 25K followers actually achieve slightly higher average engagement (3,535.84) compared to high-follower users (3,508.45). This challenges the common assumption that more followers equals more engagement and suggests content quality matters more than audience size."),
    
    ("4. Geographic Engagement Hotspots",
     "Los Angeles generates the most total engagement (1.64M), followed by Munich (1.62M) and Shanghai (1.59M). These three cities represent the Social Engine's most active user bases. Interestingly, the top 10 locations are geographically diverse, spanning 4 continents."),
    
    ("5. Massive Suspicious Sharing Activity",
     "24.3% of all posts (2,915) exhibit suspicious sharing patterns where shares exceed the combined total of likes and comments. This unusually high ratio may indicate bot-driven amplification, share-farming networks, or artificial engagement inflation. This requires further investigation."),
    
    ("6. Uniformly Distributed Engagement (Anomaly)",
     "No single user's average engagement exceeds 2x the overall average (threshold: 7,044.63). In real social media data, we would expect power-law distributions with viral outliers. This uniformity suggests the data may have been synthetically generated or heavily smoothed."),
    
    ("7. Low-Follower High-Impact Anomalies",
     "14 users with fewer than 5,000 followers rank in the top 10% of total engagement. The most anomalous is user_uerv85na from Rome (1,824 followers, 60,629 total engagement across 16 posts). These accounts may represent: organic viral creators, company-operated accounts, or engagement manipulation."),
    
    ("8. Data Corruption Impact Assessment",
     "The corrupted dataset contained 4,274 anomalous posts (34.5%): 525 with negative likes (impossible values), 1,846 with missing platforms, 1,746 with missing text content, and 663 containing HTML artifacts. Missing platform identifiers were the most common corruption pattern."),
    
    ("9. The Most Suspicious User: user_uerv85na",
     "This user appears as an anomaly across THREE separate hard-level queries (H4, H6, and implicitly H1). With only 1,824 followers but 60,629 total engagement, an average of 3,789 per post, and posts where shares exceed likes, this account represents the single most suspicious entity in the entire dataset."),
    
    ("10. Instagram is Best for Influencer Marketing",
     "Among users with 30K+ followers, Instagram delivers the highest average engagement per post (3,622.78), outperforming Reddit (3,523.90), Twitter (3,491.04), YouTube (3,488.44), and Facebook (3,441.39). For influencer marketing campaigns, Instagram should be the primary channel."),
]

for title, insight in insights:
    pdf4.set_font('Helvetica', 'B', 11)
    pdf4.cell(0, 8, safe(title), new_x="LMARGIN", new_y="NEXT")
    pdf4.set_font('Helvetica', '', 9.5)
    pdf4.multi_cell(0, 5.5, text=safe(insight), new_x="LMARGIN", new_y="NEXT")
    pdf4.ln(3)

# Conclusion
pdf4.add_page()
pdf4.set_font('Helvetica', 'B', 14)
pdf4.cell(0, 8, 'Conclusion', new_x="LMARGIN", new_y="NEXT")
pdf4.ln(2)
pdf4.set_font('Helvetica', '', 10)
pdf4.multi_cell(0, 6, text=safe(
    'Our SQL analysis of the recovered Social Engine dataset reveals a platform with balanced '
    'cross-platform presence but significant engagement anomalies. The most concerning finding is '
    'the 24.3% suspicious sharing rate, combined with 120 users who meet multiple suspicion criteria. '
    'The uniformly distributed engagement pattern (no user exceeds 2x average) is itself an anomaly '
    'that warrants investigation. We recommend the Social Engine team implement: (1) Bot detection '
    'algorithms targeting high share-to-like ratios, (2) Follower-to-engagement audit systems, and '
    '(3) Platform-specific engagement normalization to account for content format differences.'
), new_x="LMARGIN", new_y="NEXT")

pdf4.output('Phase2_Insight_Report.pdf')
print("  Phase2_Insight_Report.pdf done!")

print("\n" + "=" * 50)
print("ALL 4 SUBMISSION FILES READY:")
print("  1. SQL_Queries.pdf")
print("  2. output_Easy_E1_to_E5.jpeg")
print("     output_Medium_M1_to_M3.jpeg")
print("     output_Medium_M4_M5.jpeg")
print("     output_Hard_H1_to_H3.jpeg")
print("     output_Hard_H4_to_H6.jpeg")
print("  3. Logic_Explanation.pdf")
print("  4. Phase2_Insight_Report.pdf")
print("=" * 50)
