import json
import os

def create_cell(cell_type, source):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }

def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip()

notebook = {
    "cells": [],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5
}

# Markdown intro
intro = """# Social Engine Recovery - Round 1 (Phase 1)
**Data Vortex :: AARUUSH'26**

This notebook contains the complete workflow for recovering, cleaning, and analyzing the corrupted Social Engine dataset.
"""
notebook["cells"].append(create_cell("markdown", intro))

# Part 1: Extraction
notebook["cells"].append(create_cell("markdown", "## 1. Data Extraction\nThe dataset was extracted from the crashed Social Engine dashboard (Node 07)."))
extract_code = read_file(r"C:\Users\saish\.gemini\antigravity-ide\brain\f29b1de3-9ae7-424c-9060-f21c0039f01f\scratch\extract_data.py")
notebook["cells"].append(create_cell("code", extract_code))

# Part 2: Cleaning
notebook["cells"].append(create_cell("markdown", "## 2. Data Cleaning Pipeline\nThis pipeline cleans timestamp inconsistencies, duplicate rows, HTML artifacts, negative likes, and missing platforms."))
cleaning_code = read_file("data_cleaning.py")
notebook["cells"].append(create_cell("code", cleaning_code))

# Part 3: EDA
notebook["cells"].append(create_cell("markdown", "## 3. Exploratory Data Analysis\nGenerates 13 visualization panels and extracts key insights from the cleaned dataset."))
eda_code = read_file("eda_analysis.py")
notebook["cells"].append(create_cell("code", eda_code))

# Save notebook
with open("Social_Engine_Recovery.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Social_Engine_Recovery.ipynb created successfully!")
