# Transformer Implementation from Scratch

A complete implementation of the Transformer architecture based on "Attention Is All You Need" (Vaswani et al., 2017).

## Overview

The Transformer replaces traditional recurrent layers with self-attention mechanisms, enabling parallel computation and capturing long-range dependencies more efficiently.

## Architecture

### Core Components

| Component | Description |
|-----------|-------------|
| **PositionalEncoding** | Sinusoidal positional embeddings to encode sequence order |
| **MultiHeadAttention** | Scaled dot-product attention with multiple heads |
| **FeedForward** | Position-wise feedforward network (linear + ReLU + linear) |
| **EncoderLayer** | Self-attention + feedforward with residual connections |
| **DecoderLayer** | Self-attention + cross-attention + feedforward |

### Key Features

- **Parallel Computation**: All positions processed simultaneously
- **Constant Path Length**: O(1) operations to relate any two positions (vs linear/logarithmic in CNNs)
- **Multi-Head Attention**: Multiple attention heads capture different relationship types
- **Residual Connections**: LayerNorm after each sub-layer

## Usage

```python
from transformer import TransformerTranslator
from data_utils import Vocabulary

src_vocab = Vocabulary()
tgt_vocab = Vocabulary()

model = TransformerTranslator(src_vocab, tgt_vocab, 
    d_model=128, 
    num_heads=4, 
    num_layers=3, 
    d_ff=256)
```

## Training

```python
from transformer import train_epoch

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss(ignore_index=0)

loss = train_epoch(model, train_loader, optimizer, criterion, device)
```

## Files

| File | Description |
|------|-------------|
| `transformer.py` | Complete Transformer implementation |
| `data_utils.py` | Vocabulary and data utilities |
| `seq2seq_pytorch.py` | LSTM-based baseline for comparison |

## Key Concepts

### Why Self-Attention?

1. **Parallelization**: Compute all positions simultaneously
2. **Constant path length**: Connect any two positions in O(1) operations
3. **Interpretability**: Attention weights show word relationships

### Multi-Head Attention

Multiple attention heads allow the model to focus on different types of relationships:
- Syntactic dependencies
- Semantic connections
- Long-range correlations

### Positional Encoding

Sinusoidal functions encode position:
- `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`
- `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`

This allows the model to extrapolate to longer sequences than seen during training.

## References

- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). "Attention Is All You Need"
- Section 3.1-3.4 of the original paper for detailed mechanism descriptions
