class VariationalAutoEncoder(tf.keras.Model):
    def __init__(self, data_dim=61, latent_dim=10, beta=1.0):
        super(VariationalAutoEncoder, self).__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.beta = beta # Disentaglement Parameter for D-VAE
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.compile(optimizer='adam')

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampling([z_mean, z_log_var])
        reconstructed = self.decoder(z)
        return reconstructed, z_mean, z_log_var

    # Encoder: Data Dimension -> Latent Dimension
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

    # Decoder: Latent Dimension -> Data Dimension
    def build_decoder(self):
        # Decoder Network: Latent Dimension -> 32 -> 64 -> 128 -> Data Dimension (Reconstruct Vector)
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(32, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(latent_inputs)
        x = layers.BatchNormalization()(x)

        x = layers.Dense(64, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization()(x)

        x = layers.Dense(128, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization()(x)

        # Reconstruct the Data of given Dimension
        decoded = layers.Dense(self.data_dim, activation='sigmoid')(x)
        decoder = models.Model(latent_inputs, decoded, name="decoder")
        return decoder

    # Sample the Latent Space using Mean, STD produced by the Encoder
    def sampling(self, args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon # Mean + Standard Deviation * Epsilon
        return z

    def train_step(self, data):
        # If the data comes as a tuple, we only need the inputs.
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            # Forward pass: get reconstruction, mean, and log variance
            reconstructed, z_mean, z_log_var = self(data, training=True)

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


    # Inputs: Preprocessed Data, Data: Raw Data to label, Alpha: STD Scalar
    def find_anomalies(self, inputs, data, alpha=2):
        reconstructed_data = self(inputs)[0] # Reconstruct the data using the model
        reconstruction_error = np.mean(np.square(reconstructed_data-inputs), axis=1) # Calculate Reconstruction Error
        threshold = np.mean(reconstruction_error) + alpha*np.std(reconstruction_error) # Determine Anomlay Threshold from Distribution of Errors
        anomalies = reconstruction_error > threshold
        data['FRAUD_PRED'] = anomalies
        data_sorted = data[data['FRAUD_PRED'] == True].sort_values(by='FRAUD_PRED')
        data_sorted.to_csv('vae_results.csv')
        print("Anomaly Count: ", len(data_sorted))
