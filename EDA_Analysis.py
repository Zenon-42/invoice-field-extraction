"""
Exploratory Data Analysis (EDA) for Invoice Dataset
Generates visualizations and statistics for the hackathon submission
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class InvoiceDatasetAnalyzer:
    """Analyzes invoice extraction results and dataset characteristics"""
    
    def __init__(self, results_path: str):
        """Initialize with results JSON file"""
        with open(results_path, 'r') as f:
            self.results = json.load(f)
        
        if not isinstance(self.results, list):
            self.results = [self.results]
    
    def generate_full_report(self, output_dir: str = "eda_output"):
        """Generate comprehensive EDA report with all visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("=" * 60)
        print("GENERATING EDA REPORT")
        print("=" * 60)
        
        # 1. Processing time analysis
        self.plot_processing_time_distribution(output_path / "processing_time.png")
        
        # 2. Confidence score analysis
        self.plot_confidence_distribution(output_path / "confidence_dist.png")
        
        # 3. Cost analysis
        self.plot_cost_analysis(output_path / "cost_analysis.png")
        
        # 4. Field extraction success rates
        self.plot_field_success_rates(output_path / "field_success.png")
        
        # 5. Error analysis (if ground truth available)
        # self.plot_error_analysis(output_path / "error_analysis.png")
        
        # 6. Generate statistics table
        stats = self.generate_statistics()
        self.save_statistics_table(stats, output_path / "statistics.txt")
        
        print(f"\n✅ EDA report generated in {output_path}/")
        print("=" * 60)
    
    def plot_processing_time_distribution(self, output_path: Path):
        """Plot processing time distribution"""
        times = [r.get('processing_time_sec', 0) for r in self.results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(times, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(np.mean(times), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(times):.2f}s')
        ax1.axvline(np.median(times), color='green', linestyle='--',
                   label=f'Median: {np.median(times):.2f}s')
        ax1.set_xlabel('Processing Time (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Processing Time Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(times, vert=True)
        ax2.set_ylabel('Processing Time (seconds)')
        ax2.set_title('Processing Time Box Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Processing time analysis saved to {output_path}")
    
    def plot_confidence_distribution(self, output_path: Path):
        """Plot confidence score distribution"""
        confidences = [r.get('confidence', 0) for r in self.results]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histogram with KDE
        ax.hist(confidences, bins=20, edgecolor='black', alpha=0.6, 
               color='green', density=True, label='Histogram')
        
        # KDE overlay
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(confidences)
        x_range = np.linspace(0, 1, 100)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        
        ax.axvline(np.mean(confidences), color='blue', linestyle='--',
                  label=f'Mean: {np.mean(confidences):.2%}')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Density')
        ax.set_title('Confidence Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Confidence distribution saved to {output_path}")
    
    def plot_cost_analysis(self, output_path: Path):
        """Plot cost analysis"""
        costs = [r.get('cost_estimate_usd', 0) for r in self.results]
        total_cost = sum(costs)
        avg_cost = np.mean(costs)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Cost per document
        ax1.scatter(range(len(costs)), costs, alpha=0.6, color='purple')
        ax1.axhline(avg_cost, color='red', linestyle='--',
                   label=f'Average: ${avg_cost:.4f}')
        ax1.set_xlabel('Document Index')
        ax1.set_ylabel('Cost (USD)')
        ax1.set_title('Cost per Document')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Cumulative cost
        cumulative_cost = np.cumsum(costs)
        ax2.plot(cumulative_cost, linewidth=2, color='darkgreen')
        ax2.fill_between(range(len(cumulative_cost)), cumulative_cost, 
                         alpha=0.3, color='green')
        ax2.set_xlabel('Number of Documents')
        ax2.set_ylabel('Cumulative Cost (USD)')
        ax2.set_title(f'Cumulative Cost (Total: ${total_cost:.2f})')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Cost analysis saved to {output_path}")
    
    def plot_field_success_rates(self, output_path: Path):
        """Plot field extraction success rates"""
        field_counts = defaultdict(int)
        total_docs = len(self.results)
        
        for result in self.results:
            fields = result.get('fields', {})
            
            if fields.get('dealer_name'):
                field_counts['Dealer Name'] += 1
            if fields.get('model_name'):
                field_counts['Model Name'] += 1
            if fields.get('horse_power', 0) > 0:
                field_counts['Horse Power'] += 1
            if fields.get('asset_cost', 0) > 0:
                field_counts['Asset Cost'] += 1
            if fields.get('signature', {}).get('present'):
                field_counts['Signature'] += 1
            if fields.get('stamp', {}).get('present'):
                field_counts['Stamp'] += 1
        
        # Calculate percentages
        field_names = list(field_counts.keys())
        success_rates = [field_counts[f] / total_docs * 100 for f in field_names]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.barh(field_names, success_rates, color='coral', edgecolor='black')
        
        # Add percentage labels
        for i, (bar, rate) in enumerate(zip(bars, success_rates)):
            ax.text(rate + 1, i, f'{rate:.1f}%', va='center')
        
        ax.set_xlabel('Success Rate (%)')
        ax.set_title('Field Extraction Success Rates')
        ax.set_xlim(0, 110)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Field success rates saved to {output_path}")
    
    def generate_statistics(self) -> dict:
        """Generate comprehensive statistics"""
        times = [r.get('processing_time_sec', 0) for r in self.results]
        confidences = [r.get('confidence', 0) for r in self.results]
        costs = [r.get('cost_estimate_usd', 0) for r in self.results]
        
        stats = {
            'Total Documents': len(self.results),
            'Processing Time': {
                'Mean': f"{np.mean(times):.2f}s",
                'Median': f"{np.median(times):.2f}s",
                'Min': f"{np.min(times):.2f}s",
                'Max': f"{np.max(times):.2f}s",
                'Std Dev': f"{np.std(times):.2f}s",
                'P95': f"{np.percentile(times, 95):.2f}s",
                'P99': f"{np.percentile(times, 99):.2f}s"
            },
            'Confidence Scores': {
                'Mean': f"{np.mean(confidences):.2%}",
                'Median': f"{np.median(confidences):.2%}",
                'Min': f"{np.min(confidences):.2%}",
                'Max': f"{np.max(confidences):.2%}",
                'High Confidence (≥0.9)': sum(1 for c in confidences if c >= 0.9),
                'Low Confidence (<0.7)': sum(1 for c in confidences if c < 0.7)
            },
            'Cost Analysis': {
                'Total Cost': f"${sum(costs):.4f}",
                'Average per Document': f"${np.mean(costs):.4f}",
                'Min': f"${np.min(costs):.4f}",
                'Max': f"${np.max(costs):.4f}"
            }
        }
        
        return stats
    
    def save_statistics_table(self, stats: dict, output_path: Path):
        """Save statistics as formatted text"""
        with open(output_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("DATASET STATISTICS\n")
            f.write("=" * 60 + "\n\n")
            
            for category, values in stats.items():
                f.write(f"{category}:\n")
                f.write("-" * 40 + "\n")
                
                if isinstance(values, dict):
                    for key, value in values.items():
                        f.write(f"  {key:25s}: {value}\n")
                else:
                    f.write(f"  {values}\n")
                
                f.write("\n")
            
            f.write("=" * 60 + "\n")
        
        print(f"✓ Statistics table saved to {output_path}")
    
    def plot_error_categories(self, error_data: dict, output_path: Path):
        """Plot error analysis (requires manual error categorization)"""
        categories = list(error_data.keys())
        counts = list(error_data.values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        wedges, texts, autotexts = ax.pie(counts, labels=categories, autopct='%1.1f%%',
                                           startangle=90, colors=sns.color_palette('Set3'))
        
        ax.set_title('Error Category Distribution')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function to run EDA"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run EDA on invoice extraction results')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to results JSON file')
    parser.add_argument('--output-dir', type=str, default='eda_output',
                       help='Directory to save EDA outputs')
    
    args = parser.parse_args()
    
    analyzer = InvoiceDatasetAnalyzer(args.results)
    analyzer.generate_full_report(args.output_dir)
    
    print("\n🎉 EDA complete! Check the output directory for visualizations.")


if __name__ == "__main__":
    main()
