import pandas as pd
from typing import List, Dict, Any, Set
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class EvaluationMetrics:
    """
    Calculates and manages evaluation metrics for model performance.
    """
    def __init__(self, results_df: pd.DataFrame = None):
        """
        Initialize the evaluation metrics calculator.
        
        Args:
            results_df: DataFrame containing evaluation results
        """
        self.results_df = results_df
    
    def load_results(self, csv_path: str):
        """
        Load results from a CSV file.
        
        Args:
            csv_path: Path to CSV file containing results
        """
        self.results_df = pd.read_csv(csv_path)
    
    def calculate_basic_metrics(self) -> Dict[str, Any]:
        """
        Calculate basic performance metrics.
        
        Returns:
            Dictionary containing calculated metrics
        """
        if self.results_df is None:
            raise ValueError("Results DataFrame not initialized")
        
        metrics = {
            'total_questions': len(self.results_df),
            'correct_answers': len(self.results_df[self.results_df['is_correct']]),
            'accuracy': len(self.results_df[self.results_df['is_correct']]) / len(self.results_df) if len(self.results_df) > 0 else 0,
        }
        
        # Add average confidence if available
        if 'confidence_score' in self.results_df.columns:
            metrics['average_confidence'] = self.results_df['confidence_score'].mean()
            metrics['confidence_correct_correlation'] = self.results_df['confidence_score'].corr(self.results_df['is_correct'])
            
            # Add confidence threshold analysis
            thresholds = [0.7, 0.8, 0.9]
            for threshold in thresholds:
                high_conf = self.results_df[self.results_df['confidence_score'] >= threshold]
                if len(high_conf) > 0:
                    metrics[f'accuracy_conf_{threshold}'] = len(high_conf[high_conf['is_correct']]) / len(high_conf)
                    metrics[f'questions_above_{threshold}'] = len(high_conf)
                else:
                    metrics[f'accuracy_conf_{threshold}'] = 0
                    metrics[f'questions_above_{threshold}'] = 0
        
        return metrics
    
    def calculate_category_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate metrics by category.
        
        Returns:
            Dictionary with metrics for each category
        """
        if self.results_df is None or 'category' not in self.results_df.columns:
            raise ValueError("Results DataFrame missing or doesn't contain category information")
        
        category_metrics = {}
        for category, group in self.results_df.groupby('category'):
            category_metrics[category] = {
                'total_questions': len(group),
                'correct_answers': len(group[group['is_correct']]),
                'accuracy': len(group[group['is_correct']]) / len(group) if len(group) > 0 else 0
            }
        
        return category_metrics
    
    def save_metrics(self, output_file: str, metrics: Dict[str, Any] = None):
        """
        Save metrics to a JSON file.
        
        Args:
            output_file: Path to save metrics JSON file
            metrics: Optional metrics dict to save (if None, calculates basic metrics)
        """
        if metrics is None:
            metrics = self.calculate_basic_metrics()
        
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def plot_accuracy_by_category(self, output_file: str = None, figsize=(12, 8)):
        """
        Plot accuracy by category.
        
        Args:
            output_file: Optional file path to save the plot
            figsize: Figure size tuple
        """
        if self.results_df is None or 'category' not in self.results_df.columns:
            raise ValueError("Results DataFrame missing or doesn't contain category information")
        
        category_metrics = self.calculate_category_metrics()
        categories = list(category_metrics.keys())
        accuracies = [metrics['accuracy'] for metrics in category_metrics.values()]
        
        plt.figure(figsize=figsize)
        bars = plt.bar(categories, accuracies)
        plt.xlabel('Category')
        plt.ylabel('Accuracy')
        plt.title('Accuracy by Neurological Category')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        if output_file:
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()