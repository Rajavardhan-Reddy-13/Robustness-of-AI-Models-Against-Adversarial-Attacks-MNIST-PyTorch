# Robustness of AI Models Against Adversarial Attacks (MNIST, PyTorch)

## 📌 Project Overview
This project demonstrates how vulnerable deep learning models can be to adversarial attacks and explores defense mechanisms to improve robustness.  
We use the MNIST dataset with a Convolutional Neural Network (CNN) as the baseline classifier, then apply adversarial attacks (Gaussian noise and FGSM), followed by two defense strategies:
1. Adversarial Training (training with a mix of clean + adversarial data)
2. Variational Autoencoder (VAE) reconstruction as a denoiser  
Additionally, we implement a GAN Discriminator as an anomaly detector to identify adversarial samples.

---

## ⚙️ Setup
- Language: Python 3.x
- Frameworks: PyTorch, Torchvision
- Dataset: MNIST (28×28 grayscale digits)
- Libraries: matplotlib, numpy

### Installation
```bash
pip install torch torchvision matplotlib


---

## Part 2 — Data Preparation and Baseline CNN

```markdown
## 📂 Step 1: Data Preparation
- Load MNIST using `torchvision.datasets.MNIST`.
- Normalize to range [-1, 1] using `transforms.Normalize((0.5,), (0.5,))`.
- Split training set into:
  - 80% train
  - 20% validation
- Create DataLoaders (batch_size = 64).

---

## 🧠 Step 2: Baseline CNN Model

### Architecture
- Conv2d (1 → 32, kernel=3, padding=1) → ReLU → MaxPool(2)
- Conv2d (32 → 64, kernel=3, padding=1) → ReLU → MaxPool(2)
- Flatten (64 × 7 × 7 = 3136 features)
- FC (3136 → 128) → ReLU → Dropout(0.2)
- FC (128 → 64) → ReLU
- FC (64 → 10 logits)

### Training
- Loss: CrossEntropyLoss  
- Optimizer: Adam, lr = 0.001  
- Epochs: 10  
- Save the best model when validation loss improves.

### Results
- Test accuracy on clean MNIST: **~99%**

---

## ☁️ Step 3: Gaussian Noise Sensitivity
- Added Gaussian noise with mean=0, std=0.3 to test images.
- Evaluated CNN on noisy dataset.
- Accuracy dropped from **99% → 96%**.

---

## ⚡ Step 4: FGSM Adversarial Attack

### Implementation
1. Set `requires_grad=True` on input image.
2. Forward pass → compute CE loss.
3. Backward pass → compute gradient wrt image.
4. Perturb with ε = 0.25.
5. Clamp values to [0, 1].

### Results
- Test accuracy dropped to **~82%** under FGSM.

---

## 🛡️ Step 5: Defense 1 — Adversarial Training
- Generate adversarial examples on training set using FGSM.
- Mix **70% clean + 30% adversarial**.
- Retrain CNN with this dataset.
- Accuracy on FGSM test set: **~95%**

---

## 🔄 Step 6: Defense 2 — VAE Reconstruction

### VAE Architecture
- **Encoder**: 784 → 400 → 20 (μ, logσ²)  
- **Reparameterization**: z = μ + σ * ε  
- **Decoder**: 20 → 400 → 784 → Sigmoid reshape  

### Training
- Train only on **clean MNIST**.
- Loss: MSE reconstruction loss.

### Testing
- Pass FGSM images through VAE → reconstructed clean images.
- Classify with baseline CNN.

### Results
- Accuracy on VAE-reconstructed adversarial images: **~87%**

## 🤖 Step 7: GAN Discriminator as Anomaly Detector
- Train GAN on clean MNIST:
  - Generator: noise → fake images
  - Discriminator: real/fake classifier
- At test time:
  - Feed adversarial images into discriminator.
  - If D(x) < threshold → classify as anomaly.
- Used threshold = 0.5.

---

## 📊 Results Summary

| Scenario                        | Accuracy |
|---------------------------------|----------|
| Clean test set                  | ~99%     |
| Gaussian noise (std=0.3)        | ~96%     |
| FGSM attack (ε=0.25)            | ~82%     |
| Adversarial training defense    | ~95%     |
| VAE reconstruction defense      | ~87%     |

---

## 📝 Key Takeaways
- CNN is highly accurate on clean data but fragile to adversarial perturbations.
- Gaussian noise is less harmful than FGSM.
- Adversarial training provides the strongest defense.
- VAE helps but doesn’t fully recover clean accuracy.
- GAN Discriminator can detect anomalies and flag adversarial inputs.

---

## 🚀 Future Work
- Explore stronger attacks (PGD, CW).
- Use convolutional VAEs or diffusion models for better denoising.
- Combine defenses (adversarial training + reconstruction).

