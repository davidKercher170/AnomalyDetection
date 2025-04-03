class VariationalAutoEncoder(BaseAutoEncoder):
    def __init__(self, data_dim=61, latent_dim=10, beta=2.0):
        super().__init__(data_dim, latent_dim, name="vae")
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

    @tf.function
    def train_step(self, data):
        # If the data comes as a tuple, we only need the inputs.
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            # Forward pass: get reconstruction, mean, and log variance
            z_mean, z_log_var = self.encoder(data)
            z = self.sampling([z_mean, z_log_var])
            reconstructed = self.decoder(z)
            # Compute reconstruction loss using Mean Squared Error
            reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(data, reconstructed))

            # Multiply by the number of data dimension to scale up the loss.
            reconstruction_loss *= self.data_dim

            # Compute KL divergence loss
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            ) * self.beta

            # Total loss is the sum of the reconstruction and KL divergence losses
            total_loss = reconstruction_loss + kl_loss

        # Compute gradients and update weights
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Return a dictionary mapping metric names to current loss values.
        return {"loss": total_loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss}
