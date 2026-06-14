# ==========================================
# Iris Flower Statistical Analysis
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from functools import reduce

# ==========================================
# STEP 1: LOAD DATASET SAFELY (TRY-EXCEPT)
# ==========================================
try:
    df = pd.read_csv("Iris.csv")
except FileNotFoundError:
    print("Error: Iris.csv file not found.")
    exit()
except PermissionError:
    print("Error: Permission denied while accessing the file.")
    exit()
except Exception as e:
    print("Unexpected error:", e)
    exit()

# Handle Kaggle / UCI column differences
df.rename(columns={
    "SepalLengthCm": "sepal_length",
    "SepalWidthCm": "sepal_width",
    "PetalLengthCm": "petal_length",
    "PetalWidthCm": "petal_width",
    "Species": "species"
}, inplace=True)

# If UCI dataset without headers
if "species" not in df.columns:
    df.columns = ["sepal_length", "sepal_width",
                  "petal_length", "petal_width", "species"]

print(" File loaded successfully\n")

# ==========================================
# STEP 2: DATA DISPLAY & RELATIONAL FILTER
# ==========================================
print("First 10 rows:\n", df.head(10))
print("\nLast 5 rows:\n", df.tail(5))
print("\nData Info:")
print(df.info())

print("\nRows where sepal_length > 5.0:")
for i in range(len(df)):
    if df.loc[i, "sepal_length"] > 5.0:
        print(df.loc[i])

# ==========================================
# STEP 3: OUTLIER DETECTION USING IQR
# ==========================================
def detect_outliers(data_list):
    data = sorted(data_list)
    n = len(data)
    Q1 = data[n // 4]
    Q3 = data[(3 * n) // 4]
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = []
    for val in data:
        if val < lower or val > upper:
            outliers.append(val)
    return outliers

print("\n Outliers:")
for col in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
    print(col, ":", detect_outliers(df[col].tolist()))

# ==========================================
# STEP 4: CREATE NEW COLUMNS (WITH CHECKS)
# ==========================================

df["petal_area"] = 0.0
df["petal_to_sepal_ratio"] = 0.0
df["combined_score"] = 0.0

for i in range(len(df)):
    if df.loc[i, "sepal_length"] != 0:
        df.loc[i, "petal_area"] = (
            df.loc[i, "petal_length"] * df.loc[i, "petal_width"]
        )
        df.loc[i, "petal_to_sepal_ratio"] = (
            df.loc[i, "petal_length"] / df.loc[i, "sepal_length"]
        )
        df.loc[i, "combined_score"] = (
            df.loc[i, "sepal_length"]
            + df.loc[i, "petal_length"]
            - df.loc[i, "sepal_width"]
        )

print("\n New columns created")

# ==========================================
# STEP 5: BITWISE AND ON BINARY-ENCODED SPECIES
# ==========================================
species_map = {
    "iris-setosa": 1,
    "iris-versicolor": 2,
    "iris-virginica": 3,
    "setosa": 1,
    "versicolor": 2,
    "virginica": 3
}

encoded_species = [
    species_map.get(str(s).lower(), 0)
    for s in df["species"]
]

# FIX 1: Convert encoded values to binary form and store in a list
binary_species = [bin(x) for x in encoded_species]
print("\nBinary encoded species (first 10):", binary_species[:10])

print("\nBitwise AND (first 10 pairs):")
for i in range(10):
    a = encoded_species[i]
    b = encoded_species[i + 1]
    print(f"{a} ({bin(a)}) & {b} ({bin(b)}) = {a & b} ({bin(a & b)})")

# ==========================================
# STEP 6: SETS & UNIQUE MEASUREMENTS
# ==========================================
unique_measurements = set()
for _, row in df.iterrows():
    tup = (
        row["sepal_length"],
        row["sepal_width"],
        row["petal_length"],
        row["petal_width"]
    )
    unique_measurements.add(tup)

print("\nTotal unique measurement combinations:",
      len(unique_measurements))

print("\n10 Random Unique Tuples:")
for t in random.sample(list(unique_measurements), 10):
    print(t)

# ==========================================
# STEP 7: MANUAL STATISTICS + PANDAS
# ==========================================

# FIX 2: Recursive sum function as required by the assignment
def recursive_sum(lst):
    if not lst:
        return 0
    return lst[0] + recursive_sum(lst[1:])

def manual_mean(data):
    clean = [x for x in data if not pd.isna(x) and x != 0]
    total = recursive_sum(clean)   # uses recursion for sum
    count = len(clean)
    return total / count

def manual_median(data):
    d = sorted(data)
    n = len(d)
    if n % 2 == 0:
        return (d[n//2 - 1] + d[n//2]) / 2
    else:
        return d[n//2]

def manual_mode(data):
    freq = {}
    for x in data:
        freq[x] = freq.get(x, 0) + 1
    return max(freq, key=freq.get)

print("\n📊 Statistics:")

# Demonstrate recursive sum on sepal_length
sample = [x for x in df["sepal_length"] if x != 0 and not pd.isna(x)]
print("Recursive sum (sepal_length):", recursive_sum(sample))

for col in ["sepal_length", "sepal_width",
            "petal_length", "petal_width"]:
    clean = [x for x in df[col] if x != 0 and not pd.isna(x)]
    print(f"\n{col}")
    print("Mean  :", manual_mean(clean))
    print("Median:", manual_median(clean))
    print("Mode  :", manual_mode(clean))
    print(df[col].describe())

df.fillna(df.mean(numeric_only=True), inplace=True)

# ==========================================
# STEP 8: FUNCTIONAL PROGRAMMING
# ==========================================
petal_lengths = df["petal_length"].tolist()
mean_petal = manual_mean(petal_lengths)

filtered = list(filter(lambda x: x > mean_petal, petal_lengths))
mapped = list(map(lambda x: x * 2, filtered))
product = reduce(lambda x, y: x * y, mapped[:5])

print("\n Functional Programming:")
print("Filtered (>mean):", filtered[:10])
print("Mapped (*2):", mapped[:10])
print("Reduced product (first 5):", product)

# ==========================================
# STEP 9: SAVE REPORT (TRY-EXCEPT)
# ==========================================
try:
    with open("iris_analysis_report.txt", "w") as f:

        W = 60  # report width

        def section(title):
            f.write("\n" + "=" * W + "\n")
            f.write(f"  {title}\n")
            f.write("=" * W + "\n")

        def divider():
            f.write("-" * W + "\n")

        # ── Title Block ───────────────────────────────────────
        f.write("*" * W + "\n")
        f.write("*" + " " * (W - 2) + "*\n")
        title_line = "IRIS FLOWER STATISTICAL ANALYSIS REPORT"
        padding = (W - 2 - len(title_line)) // 2
        f.write("*" + " " * padding + title_line + " " * (W - 2 - padding - len(title_line)) + "*\n")
        f.write("*" + " " * (W - 2) + "*\n")
        f.write("*" * W + "\n")
        f.write(f"\n  Generated using Python | Dataset: Iris Flower (UCI/Kaggle)\n")
        f.write(f"  Total Records : {len(df)}  |  Features : 4  |  Classes : 3\n")

        # ── Section 1: Dataset Overview ───────────────────────
        section("SECTION 1 : DATASET OVERVIEW")
        f.write(f"  {'Total Rows':<30} : {len(df)}\n")
        f.write(f"  {'Total Columns':<30} : {len(df.columns)}\n")
        f.write(f"  {'Unique Measurement Tuples':<30} : {len(unique_measurements)}\n")
        f.write(f"  {'Species Present':<30} : {', '.join(df['species'].unique())}\n")
        f.write("\n  Rows per Species:\n")
        divider()
        f.write(f"  {'Species':<25} {'Count'}\n")
        divider()
        for sp, cnt in df["species"].value_counts().items():
            f.write(f"  {sp:<25} {cnt}\n")
        divider()

        # ── Section 2: Outlier Detection ──────────────────────
        section("SECTION 2 : OUTLIER DETECTION  (IQR METHOD)")
        f.write(f"  Formula  : Lower = Q1 - 1.5*IQR  |  Upper = Q3 + 1.5*IQR\n\n")
        divider()
        f.write(f"  {'Column':<25} {'Outlier Count':<18} {'Outlier Values'}\n")
        divider()
        for col in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
            outliers = detect_outliers(df[col].tolist())
            val_str = str(outliers) if outliers else "None"
            f.write(f"  {col:<25} {len(outliers):<18} {val_str}\n")
        divider()

        # ── Section 3: Statistics ─────────────────────────────
        section("SECTION 3 : COLUMN-WISE STATISTICS")
        for col in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
            clean = [x for x in df[col] if x != 0 and not pd.isna(x)]
            desc  = df[col].describe()
            f.write(f"\n  [ {col.upper()} ]\n")
            divider()
            f.write(f"  {'Mean   (manual)':<28} : {round(manual_mean(clean), 4)}\n")
            f.write(f"  {'Median (manual)':<28} : {round(manual_median(clean), 4)}\n")
            f.write(f"  {'Mode   (manual)':<28} : {manual_mode(clean)}\n")
            f.write(f"  {'Min':<28} : {round(desc['min'], 4)}\n")
            f.write(f"  {'Max':<28} : {round(desc['max'], 4)}\n")
            f.write(f"  {'Std Deviation':<28} : {round(desc['std'], 4)}\n")
            f.write(f"  {'25th Percentile (Q1)':<28} : {round(desc['25%'], 4)}\n")
            f.write(f"  {'75th Percentile (Q3)':<28} : {round(desc['75%'], 4)}\n")
            divider()

        # ── Section 4: New Derived Columns ────────────────────
        section("SECTION 4 : NEW DERIVED COLUMNS SUMMARY")
        divider()
        f.write(f"  {'Column':<28} {'Min':<12} {'Max':<12} {'Mean'}\n")
        divider()
        for col in ["petal_area", "petal_to_sepal_ratio", "combined_score"]:
            f.write(f"  {col:<28} {round(df[col].min(),4):<12} {round(df[col].max(),4):<12} {round(df[col].mean(),4)}\n")
        divider()

        # ── Section 5: Functional Programming ────────────────
        section("SECTION 5 : FUNCTIONAL PROGRAMMING RESULTS")
        f.write(f"  Operation : filter() -> map() -> reduce() on petal_length\n\n")
        divider()
        f.write(f"  {'Mean Petal Length':<35} : {round(mean_petal, 4)}\n")
        f.write(f"  {'Values filtered (> mean)':<35} : {len(filtered)}\n")
        f.write(f"  {'Mapped values x2 (first 5)':<35} : {mapped[:5]}\n")
        f.write(f"  {'Reduced product (first 5 mapped)':<35} : {round(product, 4)}\n")
        divider()

        # ── Section 6: Bitwise AND ────────────────────────────
        section("SECTION 6 : BITWISE AND ON ENCODED SPECIES")
        f.write(f"  Encoding : setosa=1 (0b01)  versicolor=2 (0b10)  virginica=3 (0b11)\n\n")
        divider()
        f.write(f"  {'Pair':<6} {'A':<5} {'Binary A':<12} {'B':<5} {'Binary B':<12} {'A & B':<7} {'Binary'}\n")
        divider()
        for i in range(10):
            a = encoded_species[i]
            b = encoded_species[i + 1]
            f.write(f"  {i+1:<6} {a:<5} {bin(a):<12} {b:<5} {bin(b):<12} {a & b:<7} {bin(a & b)}\n")
        divider()

        # ── Footer ────────────────────────────────────────────
        f.write("\n" + "*" * W + "\n")
        f.write("*" + " " * (W - 2) + "*\n")
        end_line = "END OF REPORT"
        ep = (W - 2 - len(end_line)) // 2
        f.write("*" + " " * ep + end_line + " " * (W - 2 - ep - len(end_line)) + "*\n")
        f.write("*" + " " * (W - 2) + "*\n")
        f.write("*" * W + "\n")

    print("\n📄 Report saved as iris_analysis_report.txt")

except Exception as e:
    print("Error while writing report:", e)

# ==========================================
# STEP 10: VISUALIZATIONS (BONUS)
# ==========================================
df[["sepal_length", "sepal_width",
    "petal_length", "petal_width"]].hist(figsize=(10, 8))
plt.suptitle("Histograms of Iris Features")
plt.savefig("histograms.png")
plt.show()

sns.boxplot(data=df[["sepal_length", "sepal_width",
                     "petal_length", "petal_width"]])
plt.title("Boxplots of Iris Features")
plt.savefig("boxplot.png")
plt.show()

sns.scatterplot(x="sepal_length", y="sepal_width",
                hue="species", data=df)
plt.title("Sepal Length vs Sepal Width")
plt.savefig("sepal_length_vs_sepal_width.png")
plt.show()

sns.scatterplot(x="petal_length", y="petal_width",
                hue="species", data=df)
plt.title("Petal Length vs Petal Width")
plt.savefig("scatterplot.png")
plt.show()