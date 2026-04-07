from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class EDAReport:
    def __init__(
        self,
        df: pd.DataFrame,
        output_dir: str = "../reports/outputs",
        selected_columns: Optional[Iterable[str]] = None,
        style: str = "whitegrid",
    ) -> None:
        self.df = df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.selected_columns = list(selected_columns) if selected_columns is not None else list(df.columns)
        self.style = style
        sns.set_style(self.style)

        self.working_df = self.df[self.selected_columns].copy()
        self.numeric_df = self.working_df.select_dtypes(include="number")

    def data_profile(self) -> pd.DataFrame:
        profile = pd.DataFrame({
            "dtype": self.working_df.dtypes.astype(str),
            "missing_count": self.working_df.isna().sum(),
            "missing_pct": self.working_df.isna().mean() * 100,
            "n_unique": self.working_df.nunique(dropna=True),
        })
        profile.to_csv(self.output_dir / "data_profile.csv")
        return profile

    def plot_numeric_distributions(self) -> None:
        for col in self.numeric_df.columns:
            plt.figure(figsize=(8, 5))
            sns.histplot(self.numeric_df[col].dropna(), kde=True)
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            plt.savefig(self.output_dir / f"dist_{col}.png")
            plt.close()

    def plot_correlation_heatmap(self) -> None:
        if self.numeric_df.shape[1] < 2:
            return

        plt.figure(figsize=(10, 8))
        corr = self.numeric_df.corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(self.output_dir / "correlation_heatmap.png")
        plt.close()

    def plot_missing_data(self) -> None:
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.working_df.isnull(), cbar=False)
        plt.title("Missing Data")
        plt.tight_layout()
        plt.savefig(self.output_dir / "missing_data.png")
        plt.close()

    def outlier_summary_iqr(self) -> pd.DataFrame:
        rows = []

        for col in self.numeric_df.columns:
            s = self.numeric_df[col].dropna()
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = ((s < lower) | (s > upper)).sum()

            rows.append({
                "column": col,
                "outliers": int(outliers)
            })

        df_out = pd.DataFrame(rows)
        df_out.to_csv(self.output_dir / "outliers.csv", index=False)
        return df_out

    def run_all(self):
        self.data_profile()
        self.plot_numeric_distributions()
        self.plot_correlation_heatmap()
        self.plot_missing_data()
        self.outlier_summary_iqr()