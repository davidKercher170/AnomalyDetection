class ContrastiveDAE(ContrastiveAutoEncoder):
  def __init__(self, data_dim=61, latent_dim=10, reconstruction_weight=1.0, contrastive_weight=0.5, noise_stddev=0.1):
        super().__init__(data_dim, latent_dim, reconstruction_weight, contrastive_weight, name="cdae")
        self.noise_stddev = noise_stddev
        self.noise_layer = layers.GaussianNoise(self.noise_stddev) # Add Noise to Inputs

  def call(self, inputs, training=True):
        noisy_inputs = self.noise_layer(inputs, training=training)
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

  def get_latent_vector(self, inputs):
        noisy_inputs = self.noise_layer(inputs)
        return self.encoder(noisy_inputs) # Get the Vectors from the Latent Dimension
