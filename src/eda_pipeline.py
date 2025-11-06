# src/eda_pipeline.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class EdaPipeline:
    def __init__(self, data_path, output_prefix="raw"):
        """
        EDA Pipeline that loads data, sets default plot style,
        and generates visual + text reports.
        """
        self.data_path = data_path
        self.output_prefix = output_prefix
        self.df = pd.read_csv(data_path)

        # Make sure output folders exist
        os.makedirs("reports/eda", exist_ok=True)
        os.makedirs("reports/figures", exist_ok=True)

        # Automatically apply default plot style
        self._set_default_plot_style()
        logging.info(f"Data loaded successfully for {output_prefix}")

    # -----------------------------
    # Default plot style
    # -----------------------------
    def _set_default_plot_style(self):
        sns.set_theme(style="whitegrid", context="talk", font_scale=1.1)
        plt.rcParams.update({
            "figure.dpi": 150,
            "figure.figsize": (10, 6),
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "axes.grid": True,
            "grid.alpha": 0.3
        })

    # -----------------------------
    # Core reports
    # -----------------------------
    def basic_info(self):
        """Generates basic info: shape, columns, dtypes, memory usage."""
        info = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "missing_count": self.df.isna().sum().to_dict(),
            "memory_usage_MB": round(self.df.memory_usage(deep=True).sum() / 1e6, 3)
        }
        logging.info(f"Basic info collected for {self.output_prefix}")
        return info

    def correlation_heatmap(self):
        """Saves a correlation heatmap of numeric features."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            self.df.corr(numeric_only=True),
            cmap="coolwarm",
            annot=True,
            fmt=".2f"
        )
        plt.title(f"Correlation Heatmap ({self.output_prefix})")
        plt.tight_layout()
        path = f"reports/figures/{self.output_prefix}_correlation_heatmap.png"
        plt.savefig(path)
        plt.close()
        logging.info(f"Correlation heatmap saved at {path}")

    # -----------------------------
    # Text summary report
    # -----------------------------
    def generate_text_summary(self):
        """Saves a simple text file with basic info + numeric stats."""
        summary_lines = [
            f"EDA Summary Report - {self.output_prefix}",
            "="*50,
            f"Shape: {self.df.shape}",
            f"Columns: {', '.join(self.df.columns)}",
            "\nColumn Types:"
        ]
        for col, dtype in self.df.dtypes.astype(str).items():
            summary_lines.append(f"  - {col}: {dtype}")

        summary_lines.append("\nMissing Values:")
        for col, missing in self.df.isna().sum().items():
            summary_lines.append(f"  - {col}: {missing}")

        summary_lines.append(f"\nMemory Usage: {round(self.df.memory_usage(deep=True).sum() / 1e6, 3)} MB")

        # Numeric summary stats
        summary_lines.append("\nNumeric Column Statistics:")
        if len(self.df.select_dtypes(include='number').columns) > 0:
            stats = self.df.describe().T
            for col, row in stats.iterrows():
                summary_lines.append(
                    f"  - {col}: count={int(row['count'])}, mean={row['mean']:.3f}, "
                    f"std={row['std']:.3f}, min={row['min']:.3f}, 25%={row['25%']:.3f}, "
                    f"50%={row['50%']:.3f}, 75%={row['75%']:.3f}, max={row['max']:.3f}"
                )
        else:
            summary_lines.append("  No numeric columns available.")

        # Top 5 rows
        summary_lines.append("\nTop 5 Rows:")
        summary_lines.extend([str(row) for row in self.df.head().to_dict(orient="records")])

        # Save to text file
        output_path = f"reports/eda/{self.output_prefix}_eda_summary.txt"
        with open(output_path, "w") as f:
            f.write("\n".join(summary_lines))

        logging.info(f"✅ Text summary report saved at {output_path}")

    # -----------------------------
    # Run full pipeline
    # -----------------------------
    def run_pipeline(self):
        """Full EDA flow — basic info, visualizations, text summary."""
        logging.info(f"🚀 Running EDA pipeline for {self.output_prefix} data...")
        info = self.basic_info()
        self.correlation_heatmap()
        self.generate_text_summary()
        logging.info(f"✅ EDA pipeline complete for {self.output_prefix}")
        return info
