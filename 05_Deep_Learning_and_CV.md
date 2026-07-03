# Deep Learning & Computer Vision

## Neural Network Fundamentals

### Forward Pass
z^[l] = W^[l] * a^[l-1] + b^[l]
a^[l] = g^[l](z^[l])

Where g is the activation function, W are weights, b are biases.

### Backpropagation
Applies chain rule to compute gradients of loss w.r.t. all parameters.

delta^[L] = dL/dz^[L] (output layer gradient)
delta^[l] = (W^[l+1]^T * delta^[l+1]) * g'^[l](z^[l])

dL/dW^[l] = delta^[l] * a^[l-1]^T
dL/db^[l] = delta^[l]

**Vanishing gradient**: gradients shrink to near-zero in deep nets via repeated multiplication by small values. Solved by ReLU, residual connections, batch normalization, careful initialization.

**Exploding gradient**: gradients grow exponentially. Solved by gradient clipping.

### Activation Functions
| Function | Formula | Range | Use |
|---|---|---|---|
| Sigmoid | 1/(1+e^(-x)) | (0,1) | Binary output |
| Tanh | (e^x - e^(-x))/(e^x + e^(-x)) | (-1,1) | Hidden layers (RNNs) |
| ReLU | max(0,x) | [0,inf) | Standard hidden layers |
| Leaky ReLU | max(0.01x, x) | (-inf,inf) | Avoids dead neurons |
| GELU | x * Phi(x) | smooth | Transformers |
| Swish | x * sigmoid(x) | smooth | Modern architectures |
| Softmax | e^(x_i) / sum(e^(x_j)) | (0,1), sums to 1 | Multiclass output |

**Dead ReLU problem**: neurons stuck at 0 if pre-activation always negative. Leaky ReLU or ELU mitigates this.

---

## Optimization

### Gradient Descent Variants
- **Batch GD**: use all data; stable but slow; memory-intensive
- **SGD**: single sample per update; noisy but fast; can escape local minima
- **Mini-batch GD**: compromise; typical batch 32-512

### SGD with Momentum
v_t = beta * v_{t-1} + (1-beta) * grad
theta = theta - alpha * v_t

Accumulates gradient direction; dampens oscillations.

### Adam (Adaptive Moment Estimation)
m_t = beta1 * m_{t-1} + (1-beta1) * grad        (first moment)
v_t = beta2 * v_{t-1} + (1-beta2) * grad^2       (second moment)
m_hat = m_t / (1-beta1^t)                         (bias correction)
v_hat = v_t / (1-beta2^t)
theta = theta - alpha * m_hat / (sqrt(v_hat) + epsilon)

Defaults: beta1=0.9, beta2=0.999, epsilon=1e-8, lr=0.001. Adapts per-parameter learning rate.

### AdamW
Adam with decoupled weight decay. Preferred for transformer training (GPT, BERT, etc.).

### Learning Rate Schedules
- **Step decay**: reduce LR by factor every N epochs
- **Cosine annealing**: LR follows cosine curve; good for fine-tuning
- **Warmup + decay**: start low, ramp up, then decay — standard for LLMs
- **ReduceLROnPlateau**: reduce when metric stops improving

---

## Regularization

### Dropout
Randomly zero out fraction p of neurons during training. At inference, scale by (1-p) or use inverted dropout. Effective regularizer; reduces co-adaptation.

### Batch Normalization
For each mini-batch: normalize layer inputs to zero mean, unit variance, then scale and shift:
y = gamma * (x - mu_batch) / sqrt(sigma^2 + epsilon) + beta

**Benefits**: faster convergence, higher LR tolerable, acts as regularizer, reduces internal covariate shift.
At inference: use running mean/variance from training.

### Layer Normalization
Normalizes across features (not batch). Used in transformers because batch size = 1 is common during generation.

### Weight Decay (L2 Regularization)
theta_t = theta_{t-1} - alpha * (grad + lambda * theta_{t-1})
Penalizes large weights; equivalent to Gaussian prior on weights.

### Early Stopping
Monitor validation loss; stop when it starts increasing. Cheap but effective.

---

## Convolutional Neural Networks (CNN)

### Convolution Operation
Output size: (W - F + 2P) / S + 1
- W = input size, F = filter size, P = padding, S = stride

**Convolution**: learnable filter slides over input, computing dot product at each position. Shares weights spatially. Detects local patterns (edges, textures, shapes).

### Key Layers
- **Conv2D**: feature extraction with learned filters
- **Pooling (Max/Avg)**: spatial downsampling; max preserves strongest activations
- **Flatten**: convert 3D feature map to 1D for FC layers
- **Global Average Pooling**: replace flatten with spatial averaging; reduces parameters, adds regularization

### CNN Architecture Progression
- **LeNet** (1998): first successful CNN; MNIST
- **AlexNet** (2012): deep CNN; ReLU; dropout; GPU training; ImageNet winner
- **VGG** (2014): deep with 3x3 filters only; simple and effective
- **ResNet** (2015): residual connections; enables 100+ layer training
- **EfficientNet** (2019): compound scaling of depth/width/resolution; used in your Animal Classification project
- **MobileNetV2** (your best result: 99.5% accuracy with 3.5M params vs ResNet50's 25.6M): depthwise separable convolutions; efficient inference

### ResNet (Used in Your Projects)
**Key idea**: skip connections (residual connections)
y = F(x, W) + x

Allows gradient to flow directly through identity shortcut. Solves vanishing gradient for very deep networks.

F(x, W) = W_2 * ReLU(W_1 * x) + x

### Depthwise Separable Convolution (MobileNet)
Standard conv: H x W x C_in x C_out multiplications per spatial location
Depthwise separable: factorize into:
1. Depthwise: H x W x C_in (one filter per input channel)
2. Pointwise (1x1 conv): C_in x C_out

~8-9x fewer operations than standard conv.

### Transfer Learning (Your Animal Classification Project)
1. Load pretrained ImageNet weights (ResNet50, EfficientNetB0, MobileNetV2)
2. Freeze base layers (feature extractor)
3. Add custom classification head
4. Fine-tune: optionally unfreeze some top layers

**Why it works**: early layers learn universal features (edges, textures); later layers learn task-specific features.

### Ensemble Methods (Your Animal Classification)
- **Homogeneous ensemble** (3x ResNet50): 94.1% — marginal gain over single model
- **Heterogeneous ensemble** (ResNet50 + EfficientNetB0 + MobileNetV2): 98.9% — diverse models capture different patterns
- **Soft voting**: average softmax probabilities across models
- Best single model: MobileNetV2 at 99.5% — efficiency >> ensembling here

---

## Diffusion Models (Your Brain MRI Project)

### Forward Process (Diffusion)
Gradually adds Gaussian noise over T timesteps:
q(x_t | x_{t-1}) = N(x_t; sqrt(1-beta_t)*x_{t-1}, beta_t*I)

Using reparameterization:
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
where alpha_bar_t = product of (1-beta_i) for i=1..t, epsilon ~ N(0,I)

### Reverse Process (Denoising)
Learned model p_theta(x_{t-1}|x_t) = N(x_{t-1}; mu_theta(x_t,t), Sigma_theta)

The model (typically U-Net) predicts the noise epsilon at each step.

**Loss**: L = E[||epsilon - epsilon_theta(x_t, t)||^2]

### Segmentation-Guided Diffusion (Your Project)
Condition on segmentation mask: p_theta(x_{t-1}|x_t, mask)

Mask is concatenated to U-Net input channels. Model learns to generate MRI consistent with the given segmentation.

### Mask-Ablated Training (MAT — Your Novel Technique)
During training, randomly zero out tumor regions in the conditioning mask. Forces model to learn to generate plausible healthy tissue without the mask guidance, enabling counterfactual tumor-removed inference.

Results: FID 219→58, KID 0.791→0.133, TumorResidual 1.625→0.546.

### DDPM vs DDIM
- **DDPM**: stochastic reverse process; ~1000 steps; slow inference
- **DDIM**: deterministic reverse; skip steps; 10-50x faster with similar quality

### Stable Diffusion Architecture
- **VAE**: encodes image to latent space (8x spatial compression)
- **U-Net**: denoises in latent space
- **Text encoder (CLIP)**: cross-attention conditions on text
- **Scheduler**: controls noise schedule (DDPM, DDIM, DPM++)

---

## Recurrent Networks

### RNN
h_t = tanh(W_h * h_{t-1} + W_x * x_t + b)

Sequential: each step depends on previous hidden state. Suffers from vanishing gradients over long sequences.

### LSTM
Gated architecture with cell state (long-term memory):
- Forget gate: f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)
- Input gate: i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)
- Cell candidate: g_t = tanh(W_g * [h_{t-1}, x_t] + b_g)
- Cell update: C_t = f_t * C_{t-1} + i_t * g_t
- Output gate: o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)
- Hidden: h_t = o_t * tanh(C_t)

### GRU
Simplified LSTM: update gate + reset gate. Fewer parameters; similar performance to LSTM.

---

## U-Net Architecture (Your Diffusion Project)

Encoder-decoder with skip connections:
- **Encoder**: repeated Conv + MaxPool; captures context
- **Bottleneck**: dense representation
- **Decoder**: UpConv + concatenate with skip; restores resolution
- **Output**: pixel-wise prediction

Skip connections preserve spatial detail lost in pooling. Standard for segmentation and diffusion backbones.

---

## Key Training Practices

### Weight Initialization
- **Xavier/Glorot**: var = 2/(n_in + n_out); for sigmoid/tanh
- **He initialization**: var = 2/n_in; for ReLU
- Poor initialization → vanishing or exploding gradients from step 1

### Gradient Clipping
Clip gradient norm to max value (typically 1.0):
if ||grad|| > clip_value: grad = grad * clip_value / ||grad||

Essential for RNNs and LLM training.

### Mixed Precision Training
See [14_Distributed_Training_and_Inference.md](14_Distributed_Training_and_Inference.md) for detailed coverage of FP16/BF16 training, loss scaling, and precision trade-offs.

### Data Augmentation (CV)
Random flip, rotation, crop, color jitter, cutout, mixup, CutMix. Increases effective dataset size; improves generalization.

---

## Interview Key Points

- **Why does batch norm help?** Reduces internal covariate shift; stabilizes training; allows higher LR; gradient flows more easily.
- **Difference between Conv and FC**: Conv shares weights spatially; translation equivariant; fewer parameters; FC is position-dependent.
- **Why residual connections?** Gradient highway bypasses nonlinearities; enables very deep networks without degradation.
- **How does dropout work at inference?** Neurons are not dropped; outputs scaled by (1-p) or inverted dropout during training eliminates need for scaling at inference.
- **When to use global average pooling vs flatten?** GAP is more regularized, fewer parameters, less prone to overfitting; flatten preserves spatial info but is large.
