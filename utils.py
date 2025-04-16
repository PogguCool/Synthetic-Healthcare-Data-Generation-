# utils.py

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

def prepare_data_for_training(synthetic_data):
    # Split data into genetic, clinical, and environmental components
    genetic_columns = ['Genetic_Variant']
    clinical_columns = ['Age', 'Gender', 'Risk_Score', 'Lab_Result_1', 'Lab_Result_2']
    environmental_columns = []  # Add environmental columns if any

    # Convert categorical variables to numerical encoding
    synthetic_data_encoded = synthetic_data.copy()
    synthetic_data_encoded['Gender'] = synthetic_data_encoded['Gender'].map({'Male': 0, 'Female': 1})
    synthetic_data_encoded['Genetic_Variant'] = synthetic_data_encoded['Genetic_Variant'].astype('category').cat.codes
    synthetic_data_encoded['Disease'] = synthetic_data_encoded['Disease'].astype('category').cat.codes

    data = {
        'genetic': synthetic_data_encoded[genetic_columns].values.astype('float32'),
        'clinical': synthetic_data_encoded[clinical_columns].values.astype('float32'),
        'environmental': synthetic_data_encoded[environmental_columns].values.astype('float32') if environmental_columns else np.empty((len(synthetic_data_encoded), 0)).astype('float32'),
    }
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.shuffle(buffer_size=1024).batch(32)
    return dataset

def train_hierarchical_vaegan(model, dataset, epochs=5):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    mse = tf.keras.losses.MeanSquaredError()

    for epoch in range(epochs):
        for batch_data in dataset:
            with tf.GradientTape() as tape:
                reconstructed_x, validity, mu, logvar = model(batch_data)
                # Flatten inputs and outputs
                real_flat = tf.concat([batch_data['genetic'], batch_data['clinical'], batch_data['environmental']], axis=1)
                reconstructed_flat = reconstructed_x
                # VAE Loss
                reconstruction_loss = mse(real_flat, reconstructed_flat)
                kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))
                vae_loss = reconstruction_loss + kl_loss
                # GAN Loss
                valid_labels = tf.ones_like(validity)
                gan_loss = bce(valid_labels, validity)
                # Total Loss
                total_loss = vae_loss + gan_loss

            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print(f"Epoch {epoch+1}, Loss: {total_loss.numpy()}")
