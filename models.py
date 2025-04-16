# models.py

import tensorflow as tf
from tensorflow.keras import layers, Model

class HierarchicalVAEGAN(Model):
    def __init__(self, input_dims, latent_dim):
        super(HierarchicalVAEGAN, self).__init__()
        # Encoder networks for different data types
        self.genetic_encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(input_dims['genetic'],)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
        ])

        self.clinical_encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(input_dims['clinical'],)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
        ])

        self.environmental_encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(input_dims['environmental'],)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
        ])

        # Latent space
        self.latent_mu = layers.Dense(latent_dim)
        self.latent_logvar = layers.Dense(latent_dim)

        # Decoder network
        self.decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(sum(input_dims.values()), activation='sigmoid'),
        ])

        # Discriminator network
        self.discriminator = tf.keras.Sequential([
            layers.InputLayer(input_shape=(sum(input_dims.values()),)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid'),
        ])

    def encode(self, x):
        z_genetic = self.genetic_encoder(x['genetic'])
        z_clinical = self.clinical_encoder(x['clinical'])
        z_environmental = self.environmental_encoder(x['environmental'])

        z = tf.concat([z_genetic, z_clinical, z_environmental], axis=1)
        mu = self.latent_mu(z)
        logvar = self.latent_logvar(z)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * logvar) * eps

    def decode(self, z):
        return self.decoder(z)

    def discriminate(self, x):
        return self.discriminator(x)

    def call(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z)
        validity = self.discriminate(reconstructed_x)
        return reconstructed_x, validity, mu, logvar
