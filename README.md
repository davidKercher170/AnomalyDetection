# AnomalyDetection
A Survey and Implementation of Modern Anomaly Detection Algorithms using Machine Learning.

# GitHub Repository README

Welcome to my GitHub repository! This project explores various autoencoder architectures with a focus on enhancing representation learning through different innovative methods. Dive into the sections below to learn more about each technique and its unique contributions. 🚀

## Table of Contents
- [Autoencoder](#autoencoder)
- [Variational Autoencoder (and Disentangled VAE)](#variational-autoencoder)
  - [Disentangled VAE](#disentangled-vae)
- [Denoising Autoencoder](#denoising-autoencoder)
- [Contrastive Autoencoder](#contrastive-autoencoder)
  - [VAE with Contrastive](#vae-with-contrastive)
  - [DAE with Contrastive](#dae-with-contrastive)
- [Additional Information](#additional-information)
- [Contributing](#contributing)
- [Contact](#contact)

---

## Autoencoder
Autoencoders are neural networks used for unsupervised learning of efficient codings, primarily for the purpose of dimensionality reduction. They work by encoding the input into a compressed representation and then decoding it to recreate the original input, which helps in learning the intrinsic structure of the data.

---

## Variational Autoencoder
Variational Autoencoders (VAEs) are probabilistic generative models that learn a latent representation of the input data by incorporating a stochastic component. Disentangled VAEs extend this approach by encouraging the latent variables to capture distinct and interpretable factors of variation, leading to more controllable generative processes.

### Disentangled VAE

---

## Denoising Autoencoder
Denoising Autoencoders (DAEs) are designed to learn robust representations by intentionally corrupting the input data and training the model to recover the original uncorrupted data. This approach forces the network to capture meaningful features and patterns in the data, improving its resilience to noise.

---

## Contrastive Autoencoder
Contrastive Autoencoders integrate contrastive learning principles into the autoencoder framework, aiming to improve feature extraction by distinguishing between similar and dissimilar data pairs. This method helps in learning more discriminative representations that are useful for downstream tasks.

### VAE with Contrastive
In this variant, a Variational Autoencoder is combined with contrastive learning to leverage the strengths of both probabilistic modeling and discriminative feature learning. The approach enhances the latent space by ensuring that similar data points are closer together while pushing apart dissimilar ones.

### DAE with Contrastive
Similarly, the Denoising Autoencoder variant with contrastive learning integrates noise robustness with the benefits of contrastive methods. This hybrid technique aims to further refine the learned representations by enforcing similarity constraints even in the presence of data corruption.

---

## Additional Information
This repository contains implementations, experiments, and visualizations related to various autoencoder models. Contributions, feedback, and discussions are highly welcome. Feel free to explore the code, open issues, or reach out if you have questions. 😊

---

## Contributing
Contributions are the heart of open source! If you have suggestions or improvements, please open an issue or submit a pull request. Follow the [contribution guidelines](CONTRIBUTING.md) for details on how to get started.

---

## Contact
For any inquiries or further discussions, please reach out via [daveek170@gmail.com](mailto:daveek170@gmail.com). Stay connected and happy coding! 💻
