class ContrastiveAutoEncoder(tf.keras.Model):
    def __init__(self, data_dim=61, latent_dim=10):
        super(ContrastiveAutoEncoder, self).__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.compile(optimizer='adam')

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

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
        latent = layers.Dense(self.latent_dim, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)

        encoder = models.Model(inputs, latent, name="encoder")
        return encoder

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

    # For VAE Model
    def sampling(self, args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon # Mean + Standard Deviation * Epsilon
        return z

    # Uses the simCLR Loss
    def contrastive_loss(self, z1, z2, temperature=0.5):
        z1 = tf.math.l2_normalize(z1, axis=1) # Normalize
        z2 = tf.math.l2_normalize(z2, axis=1) # Normalize
        similarities = tf.matmul(z1, z2, transpose_b=True) / temperature # Use the Dot Product to Get the Similarities
        labels = tf.eye(tf.shape(z1)[0]) # Diagonal Entries are the Positive Cases
        return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, similarities, from_logits=True))

    def train_step(self, x1, x2):
        with tf.GradientTape() as tape:
            z1 = self.encoder(x1) # Latent Representation 1
            z2 = self.encoder(x2) # Latent Representation 2

            # Reconstruction Loss
            y1 = self.decoder(z1) # Reconstruction 1
            reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(x1, y1)) # Compute reconstruction loss using Mean Squared Error

            contrastive_loss = self.contrastive_loss(z1, z2) # Contrastive Loss
            total_loss = reconstruction_loss + 0.5*contrastive_loss # Combine Losses

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Return a dictionary mapping metric names to current loss values.
        return total_loss

        # Inputs: Preprocessed Data, Data: Raw Data to label, Alpha: STD Scalar
    def find_anomalies(self, inputs, data, alpha=2):
        latent_vectors = self.encoder(inputs) # Get the Vectors from the Latent Dimension
        reconstructed_data = self.decoder(latent_vectors) # Reconstruct the data using the model
        reconstruction_error = np.mean(np.square(reconstructed_data-inputs), axis=1) # Calculate Reconstruction Error

        latent_center = np.mean(latent_vectors, axis=0) # Center of the Latent Vectors
        latent_distance = np.linalg.norm(latent_vectors - latent_center, axis=1) # Latent Vector Distance From Center

        recon_error_norm = (reconstruction_error - np.mean(reconstruction_error)) / np.std(reconstruction_error) # Normalize Reconstruction Error
        latent_distance_norm = (latent_distance - np.mean(latent_distance)) / np.std(latent_distance) # Normalize Latent Distance

        combined_error = recon_error_norm + latent_distance_norm # Total Error
        threshold = np.mean(combined_error) + alpha*np.std(combined_error) # Threshold for Anomalies
        anomalies = combined_error > threshold

        data['FRAUD_PRED'] = anomalies
        data_sorted = data[data['FRAUD_PRED'] == True].sort_values(by='FRAUD_PRED')
        data_sorted.to_csv('cae_results.csv')
        print("Anomaly Count: ", len(data_sorted))
