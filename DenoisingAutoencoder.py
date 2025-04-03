class DenoisingAutoEncoder(tf.keras.Model):
    def __init__(self, data_dim=61, latent_dim=10, noise_stddev=0.1):
        super(DenoisingAutoEncoder, self).__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.noise_stddev = noise_stddev
        self.noise_layer = layers.GaussianNoise(self.noise_stddev) # Add Noise to Inputs
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

    def call(self, inputs, training=True):
        noisy_inputs = self.noise_layer(inputs, training=training)
        encoded = self.encoder(noisy_inputs)
        decoded = self.decoder(encoded)
        return decoded

    # Encoder: Data Dimension -> Latent Dimension
    def build_encoder(self):
        # Encoder network: 128 -> 64 -> 32 -> Latent Dimension
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

    # Decoder: Latent Dimension -> Data Dimension
    def build_decoder(self):
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(32, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(latent_inputs)
        x = layers.BatchNormalization()(x)

        x = layers.Dense(64, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization()(x)

        x = layers.Dense(128, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization()(x)

        decoded = layers.Dense(self.data_dim, activation='sigmoid')(x)
        decoder = models.Model(latent_inputs, decoded, name="decoder")
        return decoder

    # Inputs: Preprocessed Data, Data: Raw Data to label, Alpha: STD Scalar
    def find_anomalies(self, inputs, data, alpha=2):
        reconstructed_data = self(inputs, training=False) # Reconstruct the data using the model
        reconstruction_error = np.mean(np.square(reconstructed_data-inputs), axis=1) # Calculate Reconstruction Error
        threshold = np.mean(reconstruction_error) + alpha*np.std(reconstruction_error) # Determine Anomlay Threshold from Distribution of Errors
        anomalies = reconstruction_error > threshold
        data['FRAUD_PRED'] = anomalies
        data_sorted = data[data['FRAUD_PRED'] == True].sort_values(by='FRAUD_PRED')
        data_sorted.to_csv('dae_results.csv')
        print("Anomaly Count: ", len(data_sorted))
