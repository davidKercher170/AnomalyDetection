# Anomaly Detection Algorithms
Welcome to my GitHub repository! This project explores various autoencoder architectures with a focus on enhancing representation learning through different innovative methods. Dive into the sections below to learn more about each technique and its unique contributions. 🚀

## Table of Contents
- [Autoencoder](#autoencoder)
- [Variational Autoencoder](#variational-autoencoder)
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
Autoencoders are neural networks used for unsupervised learning of efficient codings, primarily for the purpose of dimensionality reduction. They work by encoding the input into a compressed representation and then decoding it to recreate the original input, which helps in learning the intrinsic structure of the data (https://www.science.org/doi/10.1126/science.1127647). When we deconstruct and then reconstruct datapoints, we designate anomalies as datapoints that the model cannot accurately reconstruct. This is the process I use for Anomaly Detection, where I search for datapoints with reconstruction loss greater than $\alpha * \sigma$, where $\alpha \in (1,2)$ is a chosen parameter and $\sigma$ is the standard deviation of the reconstruction loss across all datapoints.

---

## Variational Autoencoder
Variational Autoencoders (VAEs) are probabilistic generative models that learn a latent representation of the input data by incorporating a stochastic component (https://arxiv.org/abs/1312.6114v10). In a VAE, instead of the encoder outputting the latent representation of the input data, the encoder outputs two values representing the mean ($\mu$) and variance ($\sigma$) for a Normal Distribution. We then sample from the produced distribution to get the latent distribution to feed into our decoder model.

### Disentangled VAE
$\beta$-Variational Autoencoders extend this approach by encouraging the latent variables to capture distinct and interpretable factors of variation, leading to more controllable generative processes (https://openreview.net/forum?id=Sy2fzU9gl). To implement this, we simply introduce the parameter $\beta \in (1,3)$ to scale the KL Loss.

---

## Denoising Autoencoder
Denoising Autoencoders (DAEs) are designed to learn robust representations by intentionally corrupting the input data and training the model to recover the original uncorrupted data. This approach forces the network to capture meaningful features and patterns in the data, improving its resilience to noise. (https://dl.acm.org/doi/abs/10.1145/1390156.1390294) In practice, we add a Gaussian Noise to the inputs before feeding into the encoder model.

---

## Contrastive Autoencoder
Contrastive Autoencoders integrate contrastive learning principles into the autoencoder framework, aiming to improve feature extraction by distinguishing between similar and dissimilar data pairs. This method helps in learning more discriminative representations that are useful for downstream tasks. To implement a Contrastive Autoencoder, we use the same autoencoder model but add a contrastive loss function. The contrastive loss looks at the difference between latent representations in our batch with a single target pair and $N-1$ negative pairs. We calculate this using the NT-Xent Loss (https://dl.acm.org/doi/pdf/10.5555/3157096.3157304) and cosine similarity $D_P$ and $D_N$ for Positive and Negative pairs:

$$L = -\mathrm{log}\frac{e^{D_P/\tau}}{\sum_{k=1}^{N} e^{D_N/\tau}}$$
$$D(u,v) = \frac{u^Tv}{\norm{u}\norm{v}}$$

with distances computed by Scaled Cosine Similarities To generate pairs, we use our initial batch of $N$ datapoints and produce and augmented set of datapoints by introducing noise. The single positive pair consists of the diagonal entries (mapping each datapoint to it's augmentation), while rest of the batch gives us our negative pairs. In research, it has been discovered that larger batch sizes typically produce stronger results. 

For this model, we can train through reconstruction loss and contrastive loss or just with contrastive loss. Additionaly, we can optionally make use of a projection head to seperate the reconstruction and contrastive losses. The projection head takes the latent representations as input and outputs vectors to feed into the contrastive loss. Seperately, the decoder decodes our initial laten representations and computes the reconstruction loss.

### VAE with Contrastive
In this variant, a Variational Autoencoder is combined with contrastive learning to leverage the strengths of both probabilistic modeling and discriminative feature learning (https://arxiv.org/abs/1902.04601). The approach enhances the latent space by ensuring that similar data points are closer together while pushing apart dissimilar ones. CVAE differs from our Contrastive Autoencoder in how the data is augmented. Rather than introducing noise to our input data, we produce our augmentations by sampling twice from the latent probability distribution. 

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
