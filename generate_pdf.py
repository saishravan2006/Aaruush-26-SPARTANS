from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Social Engine Recovery - EDA Report', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, 'Data Vortex :: AARUUSH\'26 - Round 1 Phase 1', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_report():
    pdf = PDF()
    pdf.add_page()

    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Exploratory Data Analysis (EDA)', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, 'This report contains the key insights and visualizations generated from the cleaned Social Engine dataset. The dataset includes 12,000 posts and 1,500 users.')
    pdf.ln(10)

    # Key Insights Text
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Key Insights Summary', 0, 1)
    pdf.set_font('Arial', '', 11)
    
    insights = []
    with open('eda_insights.txt', 'r', encoding='utf-8') as f:
        insights = f.readlines()
    
    for line in insights:
        line = line.strip()
        if line.startswith('[') or '💡' in line:
            clean_line = line.replace('💡', '').strip()
            clean_line = clean_line.encode('ascii', 'ignore').decode('ascii')
            pdf.multi_cell(0, 8, txt=clean_line, new_x="LMARGIN", new_y="NEXT")
    
    pdf.ln(10)

    # Add Plots
    plots_dir = 'plots'
    plots = sorted([f for f in os.listdir(plots_dir) if f.endswith('.png')])
    
    for plot in plots:
        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, plot.replace('.png', '').replace('_', ' ').title(), 0, 1, 'C')
        # Add image to fit width (leaving margins)
        pdf.image(os.path.join(plots_dir, plot), x=10, w=190)

    pdf.output('Social_Engine_EDA_Report.pdf')
    print('Social_Engine_EDA_Report.pdf created successfully!')

if __name__ == '__main__':
    create_report()
