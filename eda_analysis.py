"""Lab 4 — Descriptive Analytics: Student Performance EDA

Conduct exploratory data analysis on the student performance dataset.
Produce distribution plots, correlation analysis, hypothesis tests,
and a written findings report.

Usage:
    python eda_analysis.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_and_profile(filepath):
    """Load the dataset and generate a data profile report.

    Args:
        filepath: path to the CSV file (e.g., 'data/student_performance.csv')

    Returns:
        DataFrame: the loaded dataset

    Side effects:
        Saves a text profile to output/data_profile.txt containing:
        - Shape (rows, columns)
        - Data types for each column
        - Missing value counts per column
        - Descriptive statistics for numeric columns
    """
    os.makedirs("output", exist_ok=True)
    sns.set_style("whitegrid")
    df=pd.read_csv(filepath)
    print("df.shape:")
    print(df.shape)
    print("\ndf.info():")
    print(df.info())
    print("\ndf.describe():")
    print(df.describe(include='all'))
    print("\ndf.head():")
    print(df.head())
    missing_counts=df.isnull().sum()
    missing_percent = ((df.isnull().sum() / len(df)) * 100).round(2)
    with open("output/data_profile.txt", "w") as f:
        f.write("DATA PROFILE\n")
        f.write(f"Shape: {df.shape}\n")
        f.write(f"Data Types:\n{df.dtypes}\n")
        f.write(f"Missing Values:\n{missing_counts}\n")
        f.write(f"Missing Percentages:\n{missing_percent}\n")
        f.write(f"Descriptive Statistics:\n{df.describe(include='all')}\n")
        f.write(f"First Few Rows:\n{df.head()}\n")
    if "commit_minutes" in df.columns:
        df["commute_minutes"] = df["commute_minutes"].fillna(df["commute_minutes"].median())
    if "study_hours_weekly" in df.columns and df["study_hours_weekly"].isnull().sum() > 0:
       df = df.dropna(subset=["study_hours_weekly"])
    return df




def plot_distributions(df):
    """Create distribution plots for key numeric variables.

    Args:
        df: pandas DataFrame with the student performance data

    Returns:
        None

    Side effects:
        Saves at least 3 distribution plots (histograms with KDE or box plots)
        as PNG files in the output/ directory. Each plot should have a
        descriptive title that states what the distribution reveals.
    """
    # GPA Distribution
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    plt.figure(figsize=(8, 5))
    sns.histplot(df["gpa"], kde=True)
    plt.title("GPA Distribution")
    plt.tight_layout()
    plt.savefig("output/gpa_distribution.png")
    plt.close()
      # Study hours
    plt.figure(figsize=(8, 5))
    sns.histplot(df["study_hours_weekly"], kde=True)
    plt.title("Study Hours Weekly Distribution")
    plt.tight_layout()
    plt.savefig("output/study_hours_weekly_distribution.png")
    plt.close()

    # Attendance
    plt.figure(figsize=(8, 5))
    sns.histplot(df["attendance_pct"], kde=True)
    plt.title("Attendance Percentage Distribution")
    plt.tight_layout()
    plt.savefig("output/attendance_pct_distribution.png")
    plt.close()

    # GPA by department
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="department", y="gpa")
    plt.title("GPA by Department")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("output/gpa_by_department.png")
    plt.close()

    # Scholarship counts
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="scholarship", order=df["scholarship"].value_counts().index)
    plt.title("Scholarship Distribution")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("output/scholarship_distribution.png")
    plt.close()



    

def plot_correlations(df):
    """Analyze and visualize relationships between numeric variables.

    Args:
        df: pandas DataFrame with the student performance data

    Returns:
        None

    Side effects:
        Saves at least one correlation visualization to the output/ directory
        (e.g., a heatmap, scatter plot, or pair plot).
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()

    # Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("output/correlation_heatmap.png")
    plt.close()
    pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr_value = corr_matrix.iloc[i, j]
            pairs.append((cols[i], cols[j], corr_value))

    pairs = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)
    top_pairs = pairs[:2]

    for idx, (x_col, y_col, corr_value) in enumerate(top_pairs, start=1):
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x=x_col, y=y_col)
        plt.title(f"{x_col} vs {y_col} (r={corr_value:.2f})")
        plt.tight_layout()
        plt.savefig(f"output/scatter_{idx}_{x_col}_vs_{y_col}.png")
        plt.close()

    return corr_matrix, top_pairs


def cohen_d(group1, group2):
   
    group1 = np.array(group1)
    group2 = np.array(group2)

    pooled_std = np.sqrt(
        ((len(group1) - 1) * np.var(group1, ddof=1) + (len(group2) - 1) * np.var(group2, ddof=1))
        / (len(group1) + len(group2) - 2)
    )

    return (np.mean(group1) - np.mean(group2)) / pooled_std

def run_hypothesis_tests(df):
    """Run statistical tests to validate observed patterns.

    Args:
        df: pandas DataFrame with the student performance data

    Returns:
        dict: test results with keys like 'internship_ttest', 'dept_anova',
              each containing the test statistic and p-value

    Side effects:
        Prints test results to stdout with interpretation.

    Tests to consider:
        - t-test: Does GPA differ between students with and without internships?
        - ANOVA: Does GPA differ across departments?
        - Correlation test: Is the correlation between study hours and GPA significant?
    """
    results = {}



 

    # Hypothesis 1: internship vs GPA
    gpa_yes = df[df["has_internship"] == "Yes"]["gpa"]
    gpa_no = df[df["has_internship"] == "No"]["gpa"]

    t_stat, p_val = stats.ttest_ind(gpa_yes, gpa_no, equal_var=False)
    d = cohen_d(gpa_yes, gpa_no)

    results["internship_ttest"] = {
        "t_statistic": t_stat,
        "p_value": p_val,
        "cohens_d": d,
        "mean_yes": gpa_yes.mean(),
        "mean_no": gpa_no.mean()
    }

    print("\nHypothesis 1: Students with internships have a higher GPA")
    print(f"Mean GPA (Yes): {gpa_yes.mean():.3f}")
    print(f"Mean GPA (No): {gpa_no.mean():.3f}")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_val:.6f}")
    print(f"Cohen's d: {d:.3f}")

    if p_val < 0.05:
        print("Result: Statistically significant difference.")
    else:
        print("Result: Not statistically significant.")

    # Hypothesis 2: scholarship vs department
    contingency_table = pd.crosstab(df["department"], df["scholarship"])
    chi2, p_val2, dof, expected = stats.chi2_contingency(contingency_table)

    results["scholarship_chi2"] = {
        "chi2": chi2,
        "p_value": p_val2,
        "dof": dof
    }

    print("\nHypothesis 2: Scholarship status is associated with department")
    print(f"Chi-square statistic: {chi2:.3f}")
    print(f"p-value: {p_val2:.6f}")
    print(f"Degrees of freedom: {dof}")

    if p_val2 < 0.05:
        print("Result: Statistically significant association.")
    else:
        print("Result: No statistically significant association.")

    return results


def write_findings_report(df, top_pairs, results):
    """Write a simple findings report."""
    with open("FINDINGS.md", "w", encoding="utf-8") as f:
        f.write("# FINDINGS\n\n")

        f.write("## Dataset Description\n")
        f.write(f"- Final dataset shape: {df.shape}\n")
        f.write("- Missing values in commute_minutes were filled with the median.\n")
        f.write("- Missing values in study_hours_weekly were dropped.\n")
        f.write("- See output/data_profile.txt for full details.\n\n")

        f.write("## Distribution Findings\n")
        f.write("- GPA distribution was explored using a histogram.\n")
        f.write("- Study hours and attendance were also plotted.\n")
        f.write("- GPA differences across departments were shown using a box plot.\n")
        f.write("- Scholarship categories were shown using a bar chart.\n\n")

        f.write("## Correlation Findings\n")
        for col1, col2, corr_val in top_pairs:
            f.write(f"- {col1} and {col2}: correlation = {corr_val:.3f}\n")
        f.write("- Correlation does not imply causation.\n\n")

        f.write("## Hypothesis Test Results\n")

        ttest = results["internship_ttest"]
        f.write("### Internship and GPA\n")
        f.write(f"- Mean GPA (Yes): {ttest['mean_yes']:.3f}\n")
        f.write(f"- Mean GPA (No): {ttest['mean_no']:.3f}\n")
        f.write(f"- t-statistic: {ttest['t_statistic']:.3f}\n")
        f.write(f"- p-value: {ttest['p_value']:.6f}\n")
        f.write(f"- Cohen's d: {ttest['cohens_d']:.3f}\n\n")

        chi2 = results["scholarship_chi2"]
        f.write("### Scholarship and Department\n")
        f.write(f"- Chi-square statistic: {chi2['chi2']:.3f}\n")
        f.write(f"- p-value: {chi2['p_value']:.6f}\n")
        f.write(f"- Degrees of freedom: {chi2['dof']}\n\n")

        f.write("## Recommendations\n")
        f.write("1. Increase internship opportunities for students.\n")
        f.write("2. Support students with lower GPA through tutoring or advising.\n")
        f.write("3. Encourage better study habits because study hours may be linked to GPA.\n")


def main():
    """Run the full EDA pipeline."""
    os.makedirs("output", exist_ok=True)

    df = load_and_profile("data/student_performance.csv")
    plot_distributions(df)
    corr_matrix, top_pairs = plot_correlations(df)
    results = run_hypothesis_tests(df)
    write_findings_report(df, top_pairs, results)

    print("\nEDA complete. Files saved in output/ and FINDINGS.md created.")


if __name__ == "__main__":
    main()