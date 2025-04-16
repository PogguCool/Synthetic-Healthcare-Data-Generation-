import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance

class ModelEvaluator:
    def __init__(self, real_data, synthetic_data):
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.metrics = {}

    def evaluate_distributions(self):
        """Evaluate statistical distributions between real and synthetic data"""
        numerical_columns = ['Age', 'Death_Rate', 'Severity_Score', 'Risk_Factor']
        distribution_metrics = {}

        for col in numerical_columns:
            # Calculate Wasserstein distance
            w_distance = wasserstein_distance(
                self.real_data[col], 
                self.synthetic_data[col]
            )
            
            # Calculate KL divergence
            hist_real, _ = np.histogram(self.real_data[col], bins=50, density=True)
            hist_synthetic, _ = np.histogram(self.synthetic_data[col], bins=50, density=True)
            
            # Avoid division by zero
            hist_real = hist_real + 1e-10
            hist_synthetic = hist_synthetic + 1e-10
            
            kl_div = np.sum(hist_real * np.log(hist_real / hist_synthetic))
            
            distribution_metrics[col] = {
                'wasserstein_distance': w_distance,
                'kl_divergence': kl_div
            }
        
        self.metrics['distributions'] = distribution_metrics
        return distribution_metrics

    def evaluate_correlation_preservation(self):
        """Evaluate how well correlations are preserved"""
        real_corr = self.real_data[['Age', 'Death_Rate', 'Severity_Score', 'Risk_Factor']].corr()
        synthetic_corr = self.synthetic_data[['Age', 'Death_Rate', 'Severity_Score', 'Risk_Factor']].corr()
        
        correlation_diff = np.abs(real_corr - synthetic_corr)
        self.metrics['correlation_preservation'] = correlation_diff
        return correlation_diff

    def evaluate_feature_authenticity(self):
        """Evaluate authenticity of categorical features"""
        categorical_columns = ['Rare_Disease', 'Treatment_Response', 'Family_History']
        authenticity_metrics = {}

        for col in categorical_columns:
            real_dist = self.real_data[col].value_counts(normalize=True)
            synthetic_dist = self.synthetic_data[col].value_counts(normalize=True)
            
            # Calculate Jensen-Shannon divergence
            m = 0.5 * (real_dist + synthetic_dist)
            js_div = 0.5 * (
                np.sum(real_dist * np.log(real_dist / m)) +
                np.sum(synthetic_dist * np.log(synthetic_dist / m))
            )
            
            authenticity_metrics[col] = {
                'js_divergence': js_div,
                'category_preservation': len(synthetic_dist) / len(real_dist)
            }

        self.metrics['feature_authenticity'] = authenticity_metrics
        return authenticity_metrics

    def evaluate_disease_feature_consistency(self):
        """Evaluate consistency of disease features with medical knowledge"""
        disease_feature_consistency = {}
        
        for disease in self.synthetic_data['Rare_Disease'].unique():
            disease_data = self.synthetic_data[self.synthetic_data['Rare_Disease'] == disease]
            features_list = disease_data['Features'].str.split(', ').explode()
            feature_counts = features_list.value_counts()
            
            # Calculate feature consistency score
            consistency_score = len(feature_counts) / len(disease_data)
            disease_feature_consistency[disease] = consistency_score

        self.metrics['disease_feature_consistency'] = disease_feature_consistency
        return disease_feature_consistency

    def plot_evaluation_results(self):
        """Generate visualization of evaluation metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('Model Evaluation Metrics', fontsize=16)

        # Plot distribution metrics
        dist_df = pd.DataFrame(self.metrics['distributions']).T
        sns.barplot(data=dist_df, y=dist_df.index, x='wasserstein_distance', ax=axes[0,0])
        axes[0,0].set_title('Distribution Similarity (Wasserstein Distance)')

        # Plot correlation preservation
        sns.heatmap(self.metrics['correlation_preservation'], 
                   annot=True, cmap='coolwarm', ax=axes[0,1])
        axes[0,1].set_title('Correlation Preservation (Absolute Difference)')

        # Plot feature authenticity
        auth_df = pd.DataFrame({k: v['js_divergence'] 
                              for k, v in self.metrics['feature_authenticity'].items()}, 
                             index=[0]).melt()
        sns.barplot(data=auth_df, x='variable', y='value', ax=axes[1,0])
        axes[1,0].set_title('Feature Authenticity (JS Divergence)')

        # Plot disease feature consistency
        cons_df = pd.DataFrame(self.metrics['disease_feature_consistency'].items(), 
                             columns=['Disease', 'Consistency'])
        sns.barplot(data=cons_df, x='Consistency', y='Disease', ax=axes[1,1])
        axes[1,1].set_title('Disease-Feature Consistency')

        plt.tight_layout()
        return fig

def generate_evaluation_report(evaluator):
    """Generate a comprehensive evaluation report"""
    report = {
        'distribution_quality': {
            'mean_wasserstein': np.mean([m['wasserstein_distance'] 
                                       for m in evaluator.metrics['distributions'].values()]),
            'mean_kl_div': np.mean([m['kl_divergence'] 
                                  for m in evaluator.metrics['distributions'].values()]),
        },
        'correlation_preservation': {
            'mean_difference': evaluator.metrics['correlation_preservation'].mean().mean(),
            'max_difference': evaluator.metrics['correlation_preservation'].max().max(),
        },
        'feature_authenticity': {
            'mean_js_divergence': np.mean([m['js_divergence'] 
                                         for m in evaluator.metrics['feature_authenticity'].values()]),
            'category_preservation': np.mean([m['category_preservation'] 
                                           for m in evaluator.metrics['feature_authenticity'].values()]),
        },
        'disease_feature_consistency': {
            'mean_consistency': np.mean(list(evaluator.metrics['disease_feature_consistency'].values())),
            'min_consistency': min(evaluator.metrics['disease_feature_consistency'].values()),
            'max_consistency': max(evaluator.metrics['disease_feature_consistency'].values()),
        }
    }
    return report

# Example usage
if __name__ == "__main__":
    # Load or generate data
    from data_generation import generate_healthcare_dataset
    
    # Generate synthetic data
    synthetic_data = generate_healthcare_dataset(1000)
    real_data = generate_healthcare_dataset(1000)  # Using generated data as real for demonstration
    
    # Create evaluator
    evaluator = ModelEvaluator(real_data, synthetic_data)
    
    # Run evaluations
    evaluator.evaluate_distributions()
    evaluator.evaluate_correlation_preservation()
    evaluator.evaluate_feature_authenticity()
    evaluator.evaluate_disease_feature_consistency()
    
    # Generate and save visualization
    fig = evaluator.plot_evaluation_results()
    fig.savefig('evaluation_results.png')
    
    # Generate report
    report = generate_evaluation_report(evaluator)
    print("\nEvaluation Report:")
    for category, metrics in report.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for metric, value in metrics.items():
            print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")