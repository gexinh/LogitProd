"""
Attention-based Multiple Instance Learning (ABMIL) Model - GitHub Release Version

This module provides the AttentionGated model for multiple instance learning (MIL)
on whole slide images (WSI). The model uses attention mechanisms to aggregate
patch-level features into slide-level predictions.

Key characteristics:
  - Attention-gated mechanism for patch feature aggregation
  - Supports batch processing of variable-length patch sequences
  - Feature projection from input_dim to 512 dimensions
  - Classification head for n_classes prediction
  - Xavier initialization for linear layers

Usage example:
  from model_abmil import AttentionGated
  
  model = AttentionGated(
      input_dim=1024,
      act='relu',
      bias=False,
      dropout=False,
      args=args  # args.input_dim, args.n_classes
  )
  
  pred, Y_prob, Y_hat, _, result_dict = model(x)  # x: (batch_size, num_patches, feature_dim)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy

# -------------------------------
# Weight initialization
# -------------------------------
def initialize_weights(module):
    """
    Initialize weights for linear and layer normalization layers.
    
    Reference: CLAM (https://github.com/mahmoodlab/CLAM)
    
    Args:
        module: nn.Module - Module to initialize weights for
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            # Reference from CLAM
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

# -------------------------------
# Model class
# -------------------------------
class AttentionGated(nn.Module):
    """
    Attention-gated multiple instance learning model.
    
    This model aggregates patch-level features using an attention mechanism
    to produce slide-level predictions. The attention mechanism uses a gated
    structure with two branches (attention_a and attention_b) that are multiplied
    together before applying softmax.
    """
    def __init__(self, input_dim=1024, act='relu', bias=False, dropout=False, args=None):
        """
        Initialize AttentionGated model.

        Args:
            input_dim: int - Input feature dimension (default: 1024)
            act: str - Activation function for attention branch a ('relu', 'gelu', 'tanh') (default: 'relu')
            bias: bool - Whether to use bias in attention layers (default: False)
            dropout: bool - Whether to apply dropout in attention layers (default: False)
            args: object - Arguments object containing:
                - args.input_dim: int - Input feature dimension
                - args.n_classes: int - Number of output classes
        """
        super(AttentionGated, self).__init__()
        self.L = args.input_dim
        self.D = 128  # Attention dimension
        self.K = 1    # Number of attention heads

        # Feature projection layer
        self.feature = [nn.Linear(args.input_dim, 512)]
        self.feature += [nn.ReLU()]
        self.feature += [nn.Dropout(0.25)]
        self.feature = nn.Sequential(*self.feature)

        # Classification head
        self.classifier = nn.Sequential(  
            nn.Linear(512, args.n_classes),
        )

        # Attention branch A
        self.attention_a = [
            nn.Linear(512, self.D, bias=bias),
        ]
        if act == 'gelu': 
            self.attention_a += [nn.GELU()]
        elif act == 'relu':
            self.attention_a += [nn.ReLU()]
        elif act == 'tanh':
            self.attention_a += [nn.Tanh()]

        # Attention branch B (gating branch)
        self.attention_b = [nn.Linear(512, self.D, bias=bias),
                            nn.Sigmoid()]

        if dropout:
            self.attention_a += [nn.Dropout(0.25)]
            self.attention_b += [nn.Dropout(0.25)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        # Attention scoring layer
        self.attention_c = nn.Linear(self.D, self.K, bias=bias)

        self.apply(initialize_weights)
    def forward(self, x):
        """
        Forward pass through the attention-gated model.

        Args:
            x: torch.Tensor - Input features
                - If 3D: (batch_size, num_patches, feature_dim)
                - If 2D: (num_patches, feature_dim) or (1, num_patches, feature_dim)

        Returns:
            tuple: (pred, Y_prob, Y_hat, None, result_dict)
                - pred: torch.Tensor - Raw logits (batch_size, n_classes)
                - Y_prob: torch.Tensor - Softmax probabilities (batch_size, n_classes)
                - Y_hat: torch.Tensor - Predicted class indices (batch_size, 1)
                - None: Placeholder for slide-level features (not returned in base model)
                - result_dict: dict - Empty dictionary for additional outputs
        """
        if x.dim() == 3:  # (batch_size, num_patches, feature_dim)
            batch_size, num_patches, feature_dim = x.shape
            batch_outputs = []
            for i in range(batch_size):
                x_i = x[i]  # [num_patches, feature_dim]
                x_i = self.feature(x_i)  # [num_patches, 512]
                
                # Compute attention scores
                a = self.attention_a(x_i)  # [num_patches, 128]
                b = self.attention_b(x_i)  # [num_patches, 128]
                A = a.mul(b)               # [num_patches, 128] - gated attention
                A = self.attention_c(A)    # [num_patches, K]
                
                # Apply softmax over patches
                A = torch.transpose(A, -1, -2)  # [K, num_patches]
                A = F.softmax(A, dim=-1)        # [K, num_patches], softmax over num_patches
                
                # Weighted aggregation
                x_i = x_i * A.reshape(-1, 1)    # [num_patches, 512]
                pred = self.classifier(x_i.sum(0).unsqueeze(0))  # [1, n_classes]
                batch_outputs.append(pred)
            
            pred = torch.cat(batch_outputs, dim=0)
        else:
            # Original single sample processing
            x = self.feature(x.squeeze(0))       # [num_patches, 512]
            x = x.squeeze(0)

            # Compute attention scores
            a = self.attention_a(x)              # [num_patches, 128]
            b = self.attention_b(x)              # [num_patches, 128]
            A = a.mul(b)                         # [num_patches, 128] - gated attention
            A = self.attention_c(A)              # [num_patches, K]
            
            # Apply softmax over patches
            A = torch.transpose(A, -1, -2)       # [K, num_patches]
            A = F.softmax(A, dim=-1)             # [K, num_patches]
            
            # Weighted aggregation
            x = x * A.reshape(-1, 1)             # [num_patches, 512]
            
            pred = self.classifier(x.sum(0).unsqueeze(0))  # [1, n_classes]
            
        # Compute probabilities and predictions
        Y_prob = F.softmax(pred, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]
        
        result_dict = {}
        
        return pred, Y_prob, Y_hat, None, result_dict

