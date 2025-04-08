class ContrastiveVAE(ContrastiveAutoEncoder):
    def __init__(self, data_dim=61, latent_dim=10, reconstruction_weight=1.0, contrastive_weight=1.0, beta=1.0):
        super().__init__(data_dim, latent_dim, reconstruction_weight, contrastive_weight, name="cvae")
        self.beta = beta

    def call(self, inputs, training=True):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampling([z_mean, z_log_var])
        reconstructed = self.decoder(z)
        return reconstructed

    def build_encoder(self):
        # Encoder network: Data Dimension -> 128 -> 64 -> 32 -> Latent Dimension (mean, log_var)
        inputs = layers.Input(shape=(self.data_dim,))

        x = layers.Dense(128, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(inputs)
        x = layers.BatchNormalization()(x)

        x = layers.Dense(64, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization()(x)

        x = layers.Dense(32, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization()(x)

        # Latent Space Parameters - Create a Distribution using Predicted Mean and Standard Deviation
        mean = layers.Dense(self.latent_dim)(x)
        stddev = layers.Dense(self.latent_dim)(x)

        encoder = models.Model(inputs, [mean, stddev], name="encoder")
        return encoder

    def sampling(self, args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon # Mean + Standard Deviation * Epsilon
        return z

    def get_latent_vector(self, inputs):
        z_mean, z_log_var = self.encoder(inputs) # Get the Vectors from the Latent Dimension
        return self.sampling([z_mean, z_log_var])

    @tf.function
    def train_step(self, x1, x2):
        with tf.GradientTape() as tape:
            z1 = self.encoder(x1) # Latent Representation 1
            z_mean, z_log_var = self.encoder(x1)
            z1 = self.sampling([z_mean, z_log_var]) # First Latent Vector from Distribution

            # z2 = self.encoder(x2) # Latent Representation 1
            # z2_mean, z2_log_var = self.encoder(x2)
            z2 = self.sampling([z_mean, z_log_var]) # First Latent Vector from Distribution

            # Reconstruction Loss
            y1 = self.decoder(z1) # Reconstruction 1
            reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(x1, y1)) * self.data_dim # Compute reconstruction loss using Mean Squared Error

            # Compute KL divergence loss
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            ) * self.beta

            # Contrastive Loss
            contrastive_loss = self.contrastive_loss(z1, z2) # Contrastive Loss
            total_loss = self.reconstruction_weight*(reconstruction_loss + kl_loss) + self.contrastive_weight*contrastive_loss # Combine Losses

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Return a dictionary mapping metric names to current loss values.
        return {"loss": total_loss,
                "reconstruction_loss": self.reconstruction_weight*reconstruction_loss,
                "kl_loss": self.reconstruction_weight*kl_loss,
                "contrastive_loss": self.contrastive_weight*contrastive_loss}

    def training_loop(self, dataset, epochs):
      for epoch in range(epochs):
        epoch_losses = defaultdict(float)
        for batch in dataset:
          x1 = tf.convert_to_tensor(batch.numpy())
          x2 = tf.convert_to_tensor(batch.numpy())

          loss_dict = self.train_step(x1, x2)
          for key, value in loss_dict.items():
            epoch_losses[key] += value.numpy()

        avg_losses = {key: loss_val / len(dataset) for key, loss_val in epoch_losses.items()}
        losses_str = ", ".join([f"{key}: {avg_losses[key]:.3f}" for key in avg_losses])
        if (epoch+1) % 5 == 0:
          print(f"Epoch {epoch+1}, Losses: {losses_str}")
