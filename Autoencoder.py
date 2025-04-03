class AutoEncoder(BaseAutoEncoder):

    def build_encoder(self):
        # Encoder network: 128 -> 64 -> 32 -> Latent Dimension (mean, log_var)
        inputs = layers.Input(shape=(self.data_dim,))

        x = layers.Dense(128, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(inputs)
        x = layers.BatchNormalization()(x)

        x = layers.Dense(64, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization()(x)

        x = layers.Dense(32, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization()(x)

        latent = layers.Dense(self.latent_dim, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)

        encoder = models.Model(inputs, latent, name="encoder")
        return encoder

    @tf.function
    def train_step(self, data):
        # If the data comes as a tuple, we only need the inputs.
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            # Forward pass: get reconstruction, mean, and log variance
            reconstructed = self(data)

            # Compute reconstruction loss using Mean Squared Error
            reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(data, reconstructed))

            # Multiply by the number of data dimension to scale up the loss.
            reconstruction_loss *= self.data_dim

        # Compute gradients and update weights
        grads = tape.gradient(reconstruction_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": reconstruction_loss}
