import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class GANModel:
    def __init__(self, noise_dim, img_shape):
        self.noise_dim = noise_dim
        self.img_shape = img_shape
        
        # Build the generator and discriminator
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        # Compile the discriminator
        self.discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        # Build the GAN model
        self.gan = self.build_gan()

    def build_generator(self):
        model = models.Sequential()
        model.add(layers.Input(shape=(self.noise_dim,)))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(layers.Reshape(self.img_shape))
        return model

    def build_discriminator(self):
        model = models.Sequential()
        model.add(layers.Input(shape=self.img_shape))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    def build_gan(self):
        model = models.Sequential()
        model.add(self.generator)
        model.add(self.discriminator)
        return model

    def generate(self, num_samples):
        noise = np.random.normal(0, 1, (num_samples, self.noise_dim))
        generated_images = self.generator.predict(noise)
        return generated_images 