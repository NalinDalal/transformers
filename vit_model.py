"""
Vision Transformer (ViT) Implementation in PyTorch
Based on: "An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale"
Reference: https://github.com/google-research/vision_transformer/tree/main/vit_jax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class AddPositionEmbs(nn.Module):
    """Adds learned positional embeddings to the inputs."""
    
    def __init__(self, num_patches: int, hidden_size: int):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, hidden_size) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embedding


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""
    
    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, mlp_dim)
        self.dense2 = nn.Linear(mlp_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return x


class Encoder1DBlock(nn.Module):
    """Transformer encoder layer."""
    
    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1
    ):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(
            hidden_size, 
            num_heads, 
            dropout=attention_dropout,
            batch_first=True
        )
        self.mlp = MlpBlock(hidden_size, mlp_dim, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention block with residual
        x_norm = self.layernorm1(x)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_output)
        
        # MLP block with residual
        x_norm = self.layernorm2(x)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output
        
        return x


class Encoder(nn.Module):
    """Transformer Model Encoder."""
    
    def __init__(
        self,
        num_layers: int,
        mlp_dim: int,
        num_heads: int,
        hidden_size: int,
        num_patches: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1
    ):
        super().__init__()
        self.pos_embedding = AddPositionEmbs(num_patches, hidden_size)
        
        self.encoder_layers = nn.ModuleList([
            Encoder1DBlock(
                hidden_size=hidden_size,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                dropout=dropout,
                attention_dropout=attention_dropout
            )
            for _ in range(num_layers)
        ])
        
        self.layernorm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_embedding(x)
        
        for layer in self.encoder_layers:
            x = layer(x)
        
        return self.layernorm(x)


class VisionTransformer(nn.Module):
    """Vision Transformer model for image classification."""
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        patch_size: int = 16,
        num_classes: int = 1000,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        representation_size: Optional[int] = None,
        classifier: str = 'token'  # 'token' or 'gap'
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.classifier = classifier
        
        h, w = image_size
        num_patches = (h // patch_size) * (w // patch_size)
        patch_dim = patch_size * patch_size * 3
        
        self.patch_embedding = nn.Linear(patch_dim, hidden_size)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        
        self.encoder = Encoder(
            num_layers=num_layers,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            hidden_size=hidden_size,
            num_patches=num_patches,
            dropout=dropout,
            attention_dropout=attention_dropout
        )
        
        if representation_size is not None:
            self.pre_logits = nn.Sequential(
                nn.Linear(hidden_size, representation_size),
                nn.Tanh()
            )
        else:
            self.pre_logits = nn.Identity()
        
        self.head = nn.Linear(
            representation_size if representation_size else hidden_size, 
            num_classes
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, 0)
    
    def forward(self, x: torch.Tensor, train: bool = True) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            train: Whether in training mode
        """
        batch_size = x.shape[0]
        
        # Convert image to patches
        # (batch, 3, h, w) -> (batch, num_patches, patch_dim)
        x = self._patchify(x)
        
        # Linear projection of flattened patches
        x = self.patch_embedding(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Pass through encoder
        x = self.encoder(x)
        
        # Extract classifier token or use global average pooling
        if self.classifier == 'token':
            x = x[:, 0]  # CLS token
        elif self.classifier == 'gap':
            x = x[:, 1:].mean(dim=1)  # Global average pooling
        
        x = self.pre_logits(x)
        x = self.head(x)
        
        return x
    
    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to flattened patches."""
        batch_size, _, height, width = x.shape
        p = self.patch_size
        
        # (batch, 3, h, w) -> (batch, num_patches_h, num_patches_w, p, p, 3)
        x = x.unfold(2, p, p).unfold(3, p, p)
        
        # (batch, num_patches_h, num_patches_w, p, p, 3) -> (batch, num_patches, p*p*3)
        x = x.contiguous().view(batch_size, -1, p * p * 3)
        
        return x


class ViTConfig:
    """Configuration for different ViT variants."""
    
    @staticmethod
    def ViT_B_16(image_size: Tuple[int, int] = (224, 224), num_classes: int = 1000) -> VisionTransformer:
        return VisionTransformer(
            image_size=image_size,
            patch_size=16,
            num_classes=num_classes,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            num_layers=12,
            dropout=0.1,
            attention_dropout=0.1
        )
    
    @staticmethod
    def ViT_L_16(image_size: Tuple[int, int] = (224, 224), num_classes: int = 1000) -> VisionTransformer:
        return VisionTransformer(
            image_size=image_size,
            patch_size=16,
            num_classes=num_classes,
            hidden_size=1024,
            mlp_dim=4096,
            num_heads=16,
            num_layers=24,
            dropout=0.1,
            attention_dropout=0.1
        )
    
    @staticmethod
    def ViT_H_14(image_size: Tuple[int, int] = (224, 224), num_classes: int = 1000) -> VisionTransformer:
        return VisionTransformer(
            image_size=image_size,
            patch_size=14,
            num_classes=num_classes,
            hidden_size=1280,
            mlp_dim=5120,
            num_heads=16,
            num_layers=32,
            dropout=0.1,
            attention_dropout=0.1
        )


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test ViT model
    model = ViTConfig.ViT_B_16(num_classes=10)
    print(f"Model: ViT-B/16")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")