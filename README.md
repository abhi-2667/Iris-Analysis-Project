# Iris Flower Statistical Analysis

A Python-based exploratory data analysis (EDA) project on the classic Iris flower dataset, performing statistical analysis, feature engineering, and data visualization.

## Overview

This project loads the Iris dataset and performs a comprehensive statistical analysis using Python's data science stack, combined with manual implementations of core statistical methods to demonstrate fundamental concepts alongside standard library functions.

## Features

- **Data Loading & Cleaning** — Safe file loading with exception handling and column normalization across dataset formats (Kaggle/UCI)
- **Outlier Detection** — Identifies outliers in each feature column using the IQR (Interquartile Range) method
- **Feature Engineering** — Derives new features:
  - `petal_area` (petal length × petal width)
  - `petal_to_sepal_ratio` (petal length / sepal length)
  - `combined_score` (sepal length + petal length − sepal width)
- **Manual Statistics** — Custom implementations of mean (using recursion), median, and mode, compared against Pandas' built-in `describe()`
- **Functional Programming** — Demonstrates `filter()`, `map()`, and `reduce()` on petal length data
- **Encoding & Bitwise Operations** — Encodes species as integers and demonstrates binary conversion and bitwise AND operations
- **Set Operations** — Identifies unique measurement combinations across the dataset
- **Automated Report Generation** — Produces a formatted text report (`iris_analysis_report.txt`) summarizing all findings
- **Data Visualization** — Generates and saves:
  - Histograms of all four features
  - Boxplots for outlier visualization
  - Scatterplots (Sepal Length vs Width, Petal Length vs Width) colored by species

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Project Structure

```
Iris-Analysis-Project/
├── analysis.py                          # Main analysis script
├── Iris.csv                              # Dataset
├── iris_analysis_report.txt              # Generated statistical report
├── histograms.png                        # Feature distribution histograms
├── boxplot.png                           # Outlier detection boxplots
├── sepal_length_vs_sepal_width.png       # Sepal scatterplot
├── scatterplot.png                       # Petal scatterplot
└── README.md
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/abhi-2667/Iris-Analysis-Project.git
   cd Iris-Analysis-Project
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn
   ```

3. Run the analysis:
   ```bash
   python analysis.py
   ```

4. View outputs:
   - Statistical report: `iris_analysis_report.txt`
   - Visualizations: `histograms.png`, `boxplot.png`, `sepal_length_vs_sepal_width.png`, `scatterplot.png`

## Sample Output

The script prints dataset info, outlier counts, manual statistics (mean/median/mode), and functional programming results to the console, while saving a detailed report and visualizations to disk.

## Author

**Danaboyina Abhiram**
[GitHub: abhi-2667](https://github.com/abhi-2667)
