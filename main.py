# main.py

from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import pandas as pd
import numpy as np
from io import BytesIO
from data_generation import (
    create_healthcare_knowledge_graph, 
    generate_synthetic_data, 
    evolutionary_algorithm,
    generate_healthcare_dataset  # Add this import
)
from utils import prepare_data_for_training, train_hierarchical_vaegan
from models import HierarchicalVAEGAN
import plotly.express as px
import threading
import os
from gan_model import GANModel  # Import your GAN model
from evaluation import ModelEvaluator, generate_evaluation_report

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for session management

# Parameters for the GAN model
noise_dim = 100  # Dimension of the noise vector
img_shape = (28, 28, 1)  # Shape of the generated images (adjust as needed)

# Initialize the GAN model globally
G = GANModel(noise_dim, img_shape)  # Ensure G is an instance of GANModel

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        num_samples = int(request.form['num_samples'])
        disease_select = request.form.getlist('disease_select')
        genetic_variants = request.form.getlist('genetic_variants')

        if not disease_select or not genetic_variants:
            flash("Please select at least one disease and one genetic variant.")
            return redirect(url_for('main'))

        with app.app_context():
            # Generate synthetic data with selected diseases and variants
            synthetic_data = generate_healthcare_dataset(
                num_samples=num_samples,
                selected_diseases=disease_select,
                selected_variants=genetic_variants
            )
            synthetic_data.to_csv('synthetic_data.csv', index=False)
            
            # 1. Age & Disease - Violin Plot with Box Plot
            fig_age_disease = px.violin(synthetic_data, 
                                      x='Rare_Disease', 
                                      y='Age',
                                      color='Rare_Disease',
                                      box=True,
                                      points="all",
                                      title='Age Distribution by Disease',
                                      template='plotly_dark')
            fig_age_disease.update_layout(showlegend=False)

            # 2. Treatment & Disease - Stacked Bar Chart
            fig_treatment = px.bar(synthetic_data,
                                 x='Rare_Disease',
                                 color='Treatment_Response',
                                 title='Treatment Response by Disease',
                                 template='plotly_dark',
                                 color_discrete_map={
                                     'Positive': '#2ecc71',
                                     'Negative': '#e74c3c',
                                     'Neutral': '#3498db'
                                 })

            # 3. Features & Disease - Treemap instead of Sunburst
            features_data = synthetic_data.copy()
            features_data['Features'] = features_data['Features'].str.split(', ')
            features_data = features_data.explode('Features')
            
            # Count frequency of each feature per disease
            feature_counts = features_data.groupby(['Rare_Disease', 'Features']).size().reset_index(name='count')
            
            fig_features = px.treemap(feature_counts,
                                    path=['Rare_Disease', 'Features'],
                                    values='count',
                                    title='Disease Features Distribution',
                                    template='plotly_dark',
                                    color='count',
                                    color_continuous_scale='Viridis')

            # 4. Death Rate - Pie Chart with Multiple Traces
            death_rate_bins = pd.cut(synthetic_data['Death_Rate'], 
                                   bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                   labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            
            fig_death_rate = px.pie(values=death_rate_bins.value_counts(),
                                  names=death_rate_bins.value_counts().index,
                                  title='Death Rate Distribution',
                                  template='plotly_dark',
                                  hole=0.3)

            # 5. Family History & Disease - Grouped Bar Chart
            fig_family = px.histogram(synthetic_data,
                                    x='Rare_Disease',
                                    color='Family_History',
                                    barmode='group',
                                    title='Family History by Disease',
                                    template='plotly_dark',
                                    color_discrete_map={
                                        'Yes': '#f1c40f',
                                        'No': '#95a5a6'
                                    })

            # Update layout for all graphs
            for fig in [fig_age_disease, fig_treatment, fig_features, 
                       fig_death_rate, fig_family]:
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    title_font_size=24,
                    title_x=0.5,
                    height=500  # Make all graphs consistent height
                )

            # Convert to HTML and return
            graphs = {
                'age_disease': fig_age_disease.to_html(full_html=False),
                'treatment': fig_treatment.to_html(full_html=False),
                'features': fig_features.to_html(full_html=False),
                'death_rate': fig_death_rate.to_html(full_html=False),
                'family_history': fig_family.to_html(full_html=False)
            }

            synthetic_data_html = synthetic_data.head(50).to_html(
                classes='table table-striped table-dark', 
                index=False
            )

            return render_template('data_preview.html',
                               synthetic_data_html=synthetic_data_html,
                               synthetic_data=synthetic_data,  # Pass the full DataFrame
                               graphs=graphs)

    return render_template('index.html')

@app.route('/download/<filename>', methods=['GET'])
def download_synthetic_data(filename):
    file_path = os.path.join(os.getcwd(), filename)
    if not os.path.exists(file_path):
        flash("File not found")
        return redirect(url_for('main'))
    return send_file(file_path, as_attachment=True)

@app.route('/train_model', methods=['POST'])
def train_model():
    # Prepare data
    synthetic_data = pd.read_json(request.form['synthetic_data'])  # Assuming data is sent as JSON
    dataset = prepare_data_for_training(synthetic_data)
    
    input_dims = {
        'genetic': len([col for col in synthetic_data.columns if 'Genetic_' in col]),
        'clinical': len([col for col in synthetic_data.columns if 'Clinical_' in col]),
        'environmental': len([col for col in synthetic_data.columns if 'Environmental_' in col]),
    }
    latent_dim = 10

    # Initialize and train model
    model = HierarchicalVAEGAN(input_dims, latent_dim)
    train_thread = threading.Thread(target=train_hierarchical_vaegan, args=(model, dataset, 5))
    train_thread.start()
    train_thread.join()  # Wait for the training to complete

    flash("Model trained successfully!")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(real_data, synthetic_data)

    # Run complete evaluation
    evaluator.evaluate_distributions()
    evaluator.evaluate_correlation_preservation()
    evaluator.evaluate_feature_authenticity()
    evaluator.evaluate_disease_feature_consistency()

    # Generate visualization
    evaluator.plot_evaluation_results()

    # Get evaluation report
    report = generate_evaluation_report(evaluator)

    return redirect(url_for('main'))

if __name__ == "__main__":
    app.run(debug=True)
