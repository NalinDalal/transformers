# [AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](https://arxiv.org/pdf/2010.11929)

Since transformers work so well on texts and NLP, we tried to apply same to images.
split an image into patches and provide the sequence of linear embeddings of these patches as an input to a Transformer.

Need sufficeint amount of data to be efficient, else of no use.
Our Vision Transformer (ViT) attains excellent results when pre-trained at sufficient scale and transferred to tasks with fewer datapoints.

One efficient way is to apply in blocks, but need very high end engineering.

well one model suggest to use 2x2 size images and then extract information, applicable only to small-resolution images.

apply transformers directly to images.
split an image into patches and provide the sequence of linear embeddings of these patches as an input to a Transformer. 
Image patches are treated the same way as tokens (words) in an NLP application. 
We train the model on image classification in supervised fashion.

don't perform well on insufficient amount of data.

naive approach: pixel by pixel, quadratic complexity.

apply to local neighbours only of reach query pixel, instead of globally. => replaces convolutions.

more work interest has been shown in applying self-attention with cnn.

![Transformer-encoder](./Tranformer-encoder.png)

## **Methodology**
transformers receive a 1D sequence of token embeddings as input.
For images, convert **2D image → sequence of patches**.


Image:
$
x \in \mathbb{R}^{H \times W \times C}
$

Split into patches of size:
$
P \times P
$

Number of patches:
$
N = \frac{HW}{P^2}
$

Flatten each patch into a vector:
$
x_p \in \mathbb{R}^{N \times (P^2 \cdot C)}
$

Each patch is treated like a **token embedding**.

Linear projection converts flattened patches to model dimension (D).
$$
z_0 = x_{class}; x_p^1E; x_p^2E; \dots ; x_p^N E + E_{pos}
$$

where:
- ($E \in \mathbb{R}^{(P^2C) \times D}$) → learnable projection
- ($E_{pos} \in \mathbb{R}^{(N+1)\times D}$) → positional embeddings
- ($x_{class}$) → learnable classification token (similar to BERT CLS token)


**Transformer Encoder**

Encoder consists of repeating blocks:

1. Multi-head Self Attention (MSA)
2. Feed Forward Network (MLP)
3. LayerNorm + residual connections

$
z'*\ell = MSA(LN(z*{\ell-1})) + z_{\ell-1}
$

$
z_\ell = MLP(LN(z'*\ell)) + z'*\ell
$

Final representation comes from CLS token:
$
y = LN(z_L^0)
$

MLP uses:
GELU activation.

---

## Classification head

During pretraining:
MLP head with 1 hidden layer

During fine-tuning:
single linear layer

---

## Positional Encoding

Since transformer has no spatial awareness, positional embeddings are added: $E_{pos}$

Unlike CNNs, ViT does NOT inherently understand:

• locality
• translation equivariance
• 2D neighborhood structure

These must be learned from data.

---

## Inductive Bias

CNN bias:

* locality
* spatial hierarchy
* translation equivariance

ViT bias:

* minimal prior assumptions
* learns spatial relations from data

Result:
needs **large datasets** to perform well.

Small dataset → CNN performs better.

Large dataset → ViT surpasses CNN.

---

## Hybrid Architecture (CNN + Transformer)

Instead of raw patches:

CNN feature maps → patches → transformer

Special case:
patch size = 1 × 1
⇒ simply flatten CNN feature map.

---

## Fine-tuning at higher resolution

During fine-tuning:

patch size kept same
image resolution increased

⇒ sequence length increases.

Position embeddings interpolated in 2D to adapt.

---

## Model Variants

| Model     | Layers | Hidden dim D | Heads | Params |
| --------- | ------ | ------------ | ----- | ------ |
| ViT-Base  | 12     | 768          | 12    | 86M    |
| ViT-Large | 24     | 1024         | 16    | 307M   |
| ViT-Huge  | 32     | 1280         | 16    | 632M   |

Notation:

ViT-L/16
means:

Large model
patch size = 16 × 16

Smaller patch size:
more tokens
higher compute cost.

---

## Training setup

Optimizer:
Adam

Typical hyperparameters:
- β1 = 0.9
- β2 = 0.999
- weight decay = 0.1

Large batch training improves transfer performance.

Fine-tuning often done using SGD with momentum.

---

## Datasets used

ImageNet (1k classes)

ImageNet-21k

JFT-300M (very large dataset)

VTAB benchmark:
19 classification tasks across domains:
- Natural images
- Medical images
- Satellite imagery
- Structured tasks (counting, orientation)

---

## Key results

ViT performs best when:

trained on very large datasets
large compute available

Performance improves with scale.

Observation:

CNN better on small datasets
ViT better on large datasets

Hybrid CNN+Transformer helpful for mid-sized datasets.

---

## Complexity intuition

Naive attention on pixels: quadratic cost in number of pixels.

$O((HW)^2)$

Using patches reduces sequence length: $O(N^2)$

where:
$
N = HW/P^2
$

Patch size balances:

resolution vs compute cost.

---

## Core intuition

Image → patches → tokens → transformer.

Self-attention learns relationships between image regions globally.

CNN:
local receptive field grows layer by layer.

Transformer:
global receptive field from first layer.



