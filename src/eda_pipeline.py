# src/eda_pipeline.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import os
import logging
from IPython.display import IFrame, display

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

        # Automatically apply default plot style
        self._set_default_plot_style()
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
        logging.info(info)
        return info
    

    
    def dist_hist(self):
        """Showing Histogram Distribution of dataset"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics = ['views', 'likes', 'dislikes', 'comment_count']
        colors = ['skyblue', 'lightcoral', 'lightyellow', 'lightpink']
        titles = ['Views', 'Likes', 'Dislikes', 'Comments']
    
        for idx, (metric, color, title) in enumerate(zip(metrics, colors, titles)):
            if metric not in self.df.columns:
                continue
            row, col = divmod(idx, 2)
            axes[row, col].hist(np.log10(self.df[metric] + 1), bins=50,
                                color=color, edgecolor='black', alpha=0.7)
            axes[row, col].set_title(f'Distribution of {title} (Log Scale)')
            axes[row, col].set_xlabel(f'{title} (log10)')
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.show()
        self._save_fig(fig, "Histogram")
    
        print("\n💡 Insight: All metrics show right-skewed distributions on log scale,")
        print("   indicating most videos have moderate engagement with few viral outliers.")
    

    

    def show_box_plot(self):
        "Box plots to identify outliers"

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics = ['views', 'likes', 'dislikes', 'comment_count']
        colors = ['skyblue', 'lightcoral', 'lightyellow', 'lightpink']
        titles = ['Views', 'Likes', 'Dislikes', 'Comments']

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            row, col = idx // 2, idx % 2
            axes[row, col].boxplot(np.log10(self.df[metric] + 1), vert=True)
            axes[row, col].set_title(f'Box Plot: {title} (Log Scale)')
            axes[row, col].set_ylabel(f'{title} (log10)')
            axes[row, col].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.show()
        self._save_fig(fig, "Box")

        print("\n💡 Insight: All metrics show right-skewed distributions on log scale,")
        print("   indicating most videos have moderate engagement with few viral outliers.")
        

    def density_plot(self):
        metrics = ['views', 'likes', 'dislikes', 'comment_count']
        colors = ['skyblue', 'lightcoral', 'lightyellow', 'lightpink']
        titles = ['Views', 'Likes', 'Dislikes', 'Comments']
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for idx, (metric, color, title) in enumerate(zip(metrics, colors, titles)):
            row, col = idx // 2, idx % 2

            sns.kdeplot(np.log10(self.df[metric] + 1), ax=axes[row, col], 
                        fill=True, color=color, alpha=0.6)
            axes[row, col].set_title(f'Density Plot: {title}')
            axes[row, col].set_xlabel(f'{title} (log10)')
            axes[row, col].set_ylabel('Density')
            axes[row, col].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        self._save_fig(fig, "Density")
        


    def weekly_analysis(self):
        "Temporial Analysis"
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        # Box plot: Views by day of week
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.df, x='published_day_of_week', y='views', order=day_order, palette='Set2')
        plt.yscale('log')
        plt.title('Views Distribution by Day of Week (Log Scale)', fontsize=16, fontweight='bold')
        plt.xlabel('Day of Week')
        plt.ylabel('Views (log scale)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        fig = plt.gcf()   # get current figure object
        self._save_fig(fig, "Weekly")
        plt.show()
        




    def avg_view_day(self):
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        avg_views_by_day = self.df.groupby('published_day_of_week')['views'].mean().reindex(day_order)

        plt.figure(figsize=(12, 6))
        avg_views_by_day.plot(kind='bar', color='steelblue', edgecolor='black')
        plt.title('Average Views by Day of Week', fontsize=16, fontweight='bold')
        plt.xlabel('Day of Week')
        plt.ylabel('Average Views')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        plt.tight_layout()
        fig = plt.gcf()   # get current figure object
        self._save_fig(fig, "Day_Average")
        plt.show()
        

        print("\n💡 Insight: Analyze which day shows highest average views.")
        print(f"   Best day: {avg_views_by_day.idxmax()} with {avg_views_by_day.max():,.0f} avg views")




    def temporal_analysis(self):

        self.df['video_publish_data'] = pd.to_datetime(self.df['publish_date'])
        self.df['year_month'] = self.df['video_publish_data'].dt.to_period('M')

        # Calculate monthly averages
        monthly_stats = self.df.groupby('year_month').agg({
            'views': 'mean',
            'likes': 'mean',
            'dislikes': 'mean', 
            'comment_count': 'mean'
        }).reset_index()
        
        monthly_stats['year_month'] = monthly_stats['year_month'].astype(str)
        
        # Create time series plots
        fig, axes = plt.subplots(4, 1, figsize=(16, 14))
        
        metrics_monthly = [
            ('views', 'Average Views Over Time', '#2E86AB'),
            ('likes', 'Average Likes Over Time', '#A23B72'),
            ('dislikes', 'Average Dislikes Over Time', '#80653D'),
            ('comment_count', 'Average Comments Over Time', '#F18F01')
        ]
        
        for idx, (metric, title, color) in enumerate(metrics_monthly):
            axes[idx].plot(monthly_stats['year_month'], monthly_stats[metric], 
                           marker='o', markersize=6, linewidth=2, color=color)
            axes[idx].set_title(title, fontsize=16, fontweight='bold', pad=15)
            axes[idx].set_ylabel(f'Average {metric.replace("_", " ").title()}')
            axes[idx].grid(True, alpha=0.3, linestyle='--')
            axes[idx].tick_params(axis='x', rotation=45, labelsize=9)
            axes[idx].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
            axes[idx].xaxis.set_major_locator(plt.MaxNLocator(20))
        
        axes[-1].set_xlabel('Time Period', fontsize=12)
        plt.tight_layout()
        plt.show()
        self._save_fig(fig, "Temporal")
        
        print("\n💡 Insight: Monthly trends show how engagement metrics evolved over time.")
        print("   Look for seasonal patterns or overall growth/decline trends.")
    
    def correlation_heatmap(self):
        # Calculate correlations (using log-transformed values for better visualization)
        key_metrics = ['views', 'likes', 'dislikes', 'comment_count']
        correlation_matrix = np.log10(self.df[key_metrics] + 1).corr()
        
        # Create heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=2, fmt='.3f', cbar_kws={"shrink": 0.8})
        plt.title('Correlation Heatmap of Engagement Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        fig = plt.gcf()
        self._save_fig(fig, "Heatmap")
        plt.show()
        
        
        print("\n💡 Insight: High correlations between views and other metrics suggest")
        print("   that videos with more views tend to get proportionally more engagement.")
        print("\nCorrelation with Views:")
        for metric in ['likes', 'dislikes', 'comment_count']:
            corr = correlation_matrix.loc['views', metric]
            print(f"  - {metric.capitalize()}: {corr:.3f}")


        

        ## 6. Category Analysis
    def categorical_visuals(self):
        
        # Category analysis plots
        fig, axes = plt.subplots(4, 1, figsize=(14, 20))
        
        category_metrics = [
            ('views', 'Views by Category', '#2E86AB'),
            ('likes', 'Likes by Category', '#A23B72'),
            ('dislikes', 'Dislikes by Category', '#80653D'),
            ('comment_count', 'Comments by Category', '#F18F01')
        ]
        
        for idx, (metric, title, color) in enumerate(category_metrics):
            category_avg = self.df.groupby('category_name')[metric].mean().sort_values(ascending=False)
            
            category_avg.plot(kind='bar', ax=axes[idx], color=color, edgecolor='black')
            axes[idx].set_yscale('log')
            axes[idx].set_title(title, fontsize=16, fontweight='bold', pad=15)
            axes[idx].set_xlabel('')
            axes[idx].set_ylabel(f'Average {metric.replace("_", " ").title()} (log scale)')
            axes[idx].grid(True, alpha=0.3, axis='y')
            axes[idx].tick_params(axis='x', rotation=45, labelsize=10)
            axes[idx].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        plt.tight_layout()
        plt.show()
        self._save_fig(fig, "Categorical")
        
        # Print top categories
        print("\n=== TOP CATEGORIES BY AVERAGE VIEWS ===")
        top_categories = self.df.groupby('category_name')['views'].mean().sort_values(ascending=False).head(5)
        for cat, views in top_categories.items():
            print(f"{cat}: {views:,.0f} avg views")

    def scatter_plot(self):

        # Identify outliers using IQR method
        Q1 = self.df['views'].quantile(0.25)
        Q3 = self.df['views'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        views_outliers = self.df[(self.df['views'] < lower_bound) | (self.df['views'] > upper_bound)]
        
        print(f"\n=== OUTLIER STATISTICS ===")
        print(f"Total outliers: {len(views_outliers)} ({len(views_outliers)/len(self.df)*100:.2f}%)")
        print(f"Lower bound: {lower_bound:,.0f} views")
        print(f"Upper bound: {upper_bound:,.0f} views")
        
        # Visualize outliers
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Scatter plot
        axes[0].scatter(self.df['views'], self.df['likes'], alpha=0.3, s=10, label='Normal', color='steelblue')
        axes[0].scatter(views_outliers['views'], views_outliers['likes'], 
                        alpha=0.7, s=40, color='red', label='Outliers', edgecolor='black')
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[0].set_xlabel('Views (log scale)', fontsize=12)
        axes[0].set_ylabel('Likes (log scale)', fontsize=12)
        axes[0].set_title('Views vs Likes (Outliers Highlighted)', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(self.df['views'], vert=True)
        axes[1].set_yscale('log')
        axes[1].set_ylabel('Views (log scale)', fontsize=12)
        axes[1].set_title('Views Distribution with Outliers', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        self._save_fig(fig, "Scatter")
        
        # Percentile analysis
        percentiles = [50, 75, 90, 95, 99, 99.9]
        print("\n=== VIEWS PERCENTILE ANALYSIS ===")
        for p in percentiles:
            value = self.df['views'].quantile(p/100)
            print(f"{p}th Percentile: {value:,.0f} views")
        
        # Compare normal vs outlier characteristics
        print("\n=== OUTLIER CHARACTERISTICS ===")
        print(f"Mean views (all videos): {self.df['views'].mean():,.0f}")
        print(f"Mean views (outliers): {views_outliers['views'].mean():,.0f}")
        print(f"Median views (all videos): {self.df['views'].median():,.0f}")
        print(f"Median views (outliers): {views_outliers['views'].median():,.0f}")
        
        # Top categories for viral videos
        print("\n=== TOP CATEGORIES FOR VIRAL VIDEOS ===")
        print(views_outliers['category_name'].value_counts().head(10))

    # -----------------------------
    # Text summary report
    # -----------------------------
    import os

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
    
        # Create folder and save file
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        output_dir = os.path.join(project_root, "reports", "eda")
        os.makedirs(output_dir, exist_ok=True)  # <-- ensure folder exists
    
        output_path = os.path.join(output_dir, f"{self.output_prefix}_eda_summary.txt")  # <-- use the folder you created
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
        self.dist_hist()
        self.show_box_plot()
        self.density_plot()
        self.temporal_analysis()
        self.weekly_analysis()
        self.avg_view_day()
        self.categorical_visuals()
        self.correlation_heatmap()
        self.scatter_plot()
        self.generate_text_summary()
        self.generate_report()
        logging.info(f"✅ EDA pipeline complete for {self.output_prefix}")
        return info
    
    def _save_fig(self, fig, name):
        """Helper to save a figure in reports/figures folder."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        output_dir = os.path.join(project_root, "reports", "figures", self.output_prefix)
        os.makedirs(output_dir, exist_ok=True)

        output_path = f"{output_dir}/{name}.png"
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        logging.info(f"📊 Saved figure: {output_path}")
    


    def generate_report(self, preview_in_notebook=False):
        """
        Generate a PDF report combining visuals and text summary.
        preview_in_notebook: If True, displays the PDF in Jupyter.
        """
        # Ensure output directories exist
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        pdf_dir = os.path.join(project_root, "reports", "eda")
        fig_dir = os.path.join(project_root, "reports", "figures")
        os.makedirs(pdf_dir, exist_ok=True)
        os.makedirs(fig_dir, exist_ok=True)
    
        output_path = os.path.join(pdf_dir, f"{self.output_prefix}_eda_report.pdf")
    
        # Initialize PDF
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []
    
        # Title
        elements.append(Paragraph(f"EDA Report - {self.output_prefix}", styles['Title']))
        elements.append(Spacer(1, 0.3 * inch))
    
        # List of figures to include
        sections = [
            ("Distribution Histograms", f"{self.output_prefix}/Histogram.png"),
            ("Box Plots", f"{self.output_prefix}/Box.png"),
            ("Density Plots", f"{self.output_prefix}/Density.png"),
            ("Weekly Views", f"{self.output_prefix}/Weekly.png"),
            ("Average Views by Day", f"{self.output_prefix}/Day_Average.png"),
            ("Temporal Trends", f"{self.output_prefix}/Temporal.png"),
            ("Category Insights", f"{self.output_prefix}/Categorical.png"),
            ("Correlation Heatmap", f"{self.output_prefix}/Heatmap.png"),
            ("Outlier & Scatter Analysis", f"{self.output_prefix}/Scatter.png"),
        ]
    
        # Add figures if they exist
        for title, filename in sections:
            path = os.path.join(fig_dir, filename)
            if os.path.exists(path):
                elements.append(Paragraph(title, styles['Heading2']))
                elements.append(Image(path, width=6.5*inch, height=4*inch))
                elements.append(Spacer(1, 0.3 * inch))
            else:
                logging.warning(f"⚠️ Figure not found and skipped: {path}")
    
        # Add footer text
        elements.append(Paragraph("Generated automatically by EdaPipeline.", styles['Normal']))
    
        # Build PDF
        doc.build(elements)
        logging.info(f"📘 PDF Report saved: {output_path}")
    
        # Preview in Jupyter if requested
        if preview_in_notebook:
            display(IFrame(output_path, width=800, height=600))