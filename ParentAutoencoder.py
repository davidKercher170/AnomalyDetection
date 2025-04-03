from abc import ABC, abstractmethod
class BaseAutoEncoder(tf.keras.Model, ABC):
    def __init__(self, data_dim=61, latent_dim=16, name="ae"):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.compile(optimizer='adam')
        self.name=name

    def call(self, inputs, training=True):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

    @abstractmethod
    def build_encoder(self):
        """Construct and return the encoder model."""
        pass

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

    @abstractmethod
    def calculate_anomalies(self, inputs, alpha=2):
        """
        Compute the reconstruction error and flag anomalies based on the error exceeding a threshold.
        """
        pass

    def calculate_anomalies(self, inputs, alpha=2):
        """
        Compute the reconstruction error and flag anomalies based on the error exceeding a threshold.
        """
        reconstructed_data = self(inputs, training=False)[0] # Reconstruct the data using the model
        reconstruction_error = np.mean(np.square(reconstructed_data-inputs), axis=1) # Calculate Reconstruction Error
        threshold = np.mean(reconstruction_error) + alpha*np.std(reconstruction_error) # Determine Anomlay Threshold from Distribution of Errors
        anomalies = reconstruction_error > threshold
        return anomalies

    def find_anomalies(self, inputs, data_model, alpha=2):
        anomalies = self.calculate_anomalies(inputs, alpha) # Calculate Anomalies
        data_model['FRAUD_PRED'] = anomalies # Label Data
        data_model = data_model[data_model['FRAUD_PRED'] == True] # Filter to Predicted Labels
        data_model.to_csv(self.name+'_results.csv') # Save Predicted Anomalies as CSV
        print("Anomaly Count:", len(data_model))

    def run_model(self, data, epochs=10):
      features_dict = {name: np.array(value) for name, value in train.items()}
      preprocessed_features = model_preprocessing(features_dict)  # Preprocess the input features
      data_model=data.copy()
      self.fit(preprocessed_features, preprocessed_features, epochs=epochs, batch_size=384)
      self.find_anomalies(preprocessed_features, data_model)
