from collections import defaultdict

class ContrastiveAutoEncoder(AutoEncoder):
    def __init__(self, data_dim=61, latent_dim=10, reconstruction_weight=1.0, contrastive_weight=0.5, name="cae"):
        super().__init__(data_dim, latent_dim, name)
        self.reconstruction_weight = reconstruction_weight
        self.contrastive_weight = contrastive_weight # For Contrastive Loss
        if reconstruction_weight==0.0: self.name = self.name+"_no_recon"

    # Uses the simCLR Loss
    def contrastive_loss_old(self, z1, z2, temperature=0.07):
        z1 = tf.math.l2_normalize(z1, axis=1) # Normalize
        z2 = tf.math.l2_normalize(z2, axis=1) # Normalize
        similarities = tf.matmul(z1, z2, transpose_b=True) / temperature # Use the Dot Product to Get the Similarities
        labels = tf.eye(tf.shape(z1)[0]) # Diagonal Entries are the Positive Cases
        return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, similarities, from_logits=True))

    def contrastive_loss(self, z1, z2, temperature=0.07):
        """
        Compute the Normalized Temperature-scaled Cross Entropy (NT-Xent) loss.

        Args:
            z1, z2: latent representations for two augmented views, shape (batch_size, latent_dim)
            temperature: scaling hyperparameter.

        Returns:
            Scalar loss value.
        """
        # Normalize the latent vectors along the feature dimension.
        z1 = tf.math.l2_normalize(z1, axis=1)
        z2 = tf.math.l2_normalize(z2, axis=1)

        # Compute similarity logits as scaled cosine similarities.
        logits = tf.matmul(z1, z2, transpose_b=True) / temperature  # shape: [batch_size, batch_size]

        # The positive pairs are along the diagonal.
        batch_size = tf.shape(logits)[0]
        labels = tf.range(batch_size)

        # Compute the cross-entropy loss. For each row, the positive sample is at the index that matches the row.
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

        # Return the mean loss over the batch.
        return tf.reduce_mean(loss)


    @tf.function
    def train_step(self, x1, x2):
        with tf.GradientTape() as tape:
            z1 = self.get_latent_vector(x1) # Latent Representation 1
            z2 = self.get_latent_vector(x2) # Latent Representation 2

            # Reconstruction Loss
            y1 = self.decoder(z1) # Reconstruction 1
            reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(x1, y1))*self.data_dim # Compute reconstruction loss using Mean Squared Error

            contrastive_loss = self.contrastive_loss(z1, z2) # Contrastive Loss
            total_loss = self.reconstruction_weight*reconstruction_loss + self.contrastive_weight*contrastive_loss # Combine Losses

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Return a dictionary mapping metric names to current loss values.
        return {"loss": total_loss,
                "reconstruction_loss": self.reconstruction_weight*reconstruction_loss,
                "contrastive_loss": self.contrastive_weight*contrastive_loss}

    def get_latent_vector(self, inputs):
        return self.encoder(inputs) # Get the Vectors from the Latent Dimension

        # Inputs: Preprocessed Data, Data: Raw Data to label, Alpha: STD Scalar
    def calculate_anomalies(self, inputs, alpha=2):
        latent_vectors = self.get_latent_vector(inputs) # Get the Vectors from the Latent Dimension
        reconstructed_data = self.decoder(latent_vectors) # Reconstruct the data using the model
        reconstruction_error = np.mean(np.square(reconstructed_data-inputs), axis=1) # Calculate Reconstruction Error

        latent_center = np.mean(latent_vectors, axis=0) # Center of the Latent Vectors
        latent_distance = np.linalg.norm(latent_vectors - latent_center, axis=1) # Latent Vector Distance From Center

        recon_error_norm = (reconstruction_error - np.mean(reconstruction_error)) / np.std(reconstruction_error) # Normalize Reconstruction Error
        latent_distance_norm = (latent_distance - np.mean(latent_distance)) / np.std(latent_distance) # Normalize Latent Distance

        combined_error = self.reconstruction_weight*recon_error_norm + self.contrastive_weight*latent_distance_norm # Total Weighted Error
        threshold = np.mean(combined_error) + alpha*np.std(combined_error) # Threshold for Anomalies
        anomalies = combined_error > threshold

        return anomalies

    def training_loop(self, dataset, epochs):
      for epoch in range(epochs):
        epoch_losses = defaultdict(float)
        for batch in dataset:
          x1 = self.augment_data(batch.numpy())
          x2 = self.augment_data(batch.numpy()) # Positive/Negative Pairs (for each datapoint in batch)
          x1 = tf.convert_to_tensor(x1)
          x2 = tf.convert_to_tensor(x2)

          loss_dict = self.train_step(x1, x2)
          for key, value in loss_dict.items():
            epoch_losses[key] += value.numpy()

        avg_losses = {key: loss_val / len(dataset) for key, loss_val in epoch_losses.items()}
        losses_str = ", ".join([f"{key}: {avg_losses[key]:.3f}" for key in avg_losses])
        if (epoch+1) % 5 == 0:
          print(f"Epoch {epoch+1}, Losses: {losses_str}")

    def augment_data(self, x, stddev=0.05):
        x_noisy = self.noise_data(x, stddev)
        x_dropout = self.dropout_data(x_noisy)
        return x_dropout

    def noise_data(self, x, stddev=0.005):
      noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=stddev)
      return tf.clip_by_value(x + noise, 0., 1.)

    def dropout_data(self, x, dropout_rate=0.1):
      # 3. Dropout: randomly drop features (set to zero) per sample
      dropout_mask = np.random.binomial(1, 1 - dropout_rate, size=x.shape)
      return x * dropout_mask

    def run_model(self, data, epochs=50, batch_size=512, buffer_size=5000):
      data_model=data.copy()
      features_dict = {name: np.array(value) for name, value in train.items()}
      preprocessed_features = model_preprocessing(features_dict)  # Preprocess the input features
      dataset = tf.data.Dataset.from_tensor_slices(preprocessed_features)
      dataset = dataset.shuffle(buffer_size).batch(batch_size)
      self.training_loop(dataset, epochs)
      self.find_anomalies(preprocessed_features, data_model)
