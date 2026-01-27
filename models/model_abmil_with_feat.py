"""
Attention-based Multiple Instance Learning (ABMIL) Model with Feature Extraction - GitHub Release Version

This module provides the AttentionGatedWithFeat model, which extends the base
AttentionGated model to also return slide-level aggregated features. This is useful
for downstream tasks that require intermediate feature representations.

Key characteristics:
  - Inherits from AttentionGated model
  - Returns slide-level features (512-dim) in addition to predictions
  - Supports batch processing of variable-length patch sequences
  - Same attention-gated mechanism as base model

Usage example:
  from model_abmil_with_feat import AttentionGatedWithFeat
  
  model = AttentionGatedWithFeat(
      input_dim=1024,
      act='relu',
      bias=False,
      dropout=False,
      args=args  # args.input_dim, args.n_classes
  )
  
  pred, Y_prob, Y_hat, wsi_feature, result_dict = model(x)  # x: (batch_size, num_patches, feature_dim)
  # wsi_feature: (batch_size, 512) - aggregated slide-level features
"""
import torch
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_abmil import AttentionGated

# -------------------------------
# Model class
# -------------------------------
class AttentionGatedWithFeat(AttentionGated):
    """
    Attention-gated multiple instance learning model with feature extraction.
    
    This model extends AttentionGated to also return slide-level aggregated features.
    The aggregated features can be used for downstream tasks or saved for later use.
    """
    def forward(self, x):
        """
        Forward pass through the attention-gated model with feature extraction.

        Args:
            x: torch.Tensor - Input features
                - If 3D: (batch_size, num_patches, feature_dim)
                - If 2D: (num_patches, feature_dim) or (1, num_patches, feature_dim)

        Returns:
            tuple: (pred, Y_prob, Y_hat, wsi_feature, result_dict)
                - pred: torch.Tensor - Raw logits (batch_size, n_classes)
                - Y_prob: torch.Tensor - Softmax probabilities (batch_size, n_classes)
                - Y_hat: torch.Tensor - Predicted class indices (batch_size, 1)
                - wsi_feature: torch.Tensor - Aggregated slide-level features (batch_size, 512)
                - result_dict: dict - Empty dictionary for additional outputs
        """
        if x.dim() == 3:  # (batch_size, num_patches, feature_dim)
            batch_size, num_patches, feature_dim = x.shape
            batch_outputs = []
            batch_features = []
            for i in range(batch_size):
                x_i = x[i]  # (num_patches, feature_dim)
                x_i = self.feature(x_i)  # (num_patches, 512)
                
                # Compute attention scores
                a = self.attention_a(x_i)
                b = self.attention_b(x_i)
                A = a.mul(b)  # Gated attention
                A = self.attention_c(A)
                A = torch.transpose(A, -1, -2)  # [K, num_patches]
                A = F.softmax(A, dim=-1)  # Softmax over patches
                
                # Weighted aggregation
                x_weighted = x_i * A.reshape(-1, 1)
                wsi_feature = x_weighted.sum(0)  # (512,) - aggregated feature
                pred = self.classifier(wsi_feature.unsqueeze(0))
                batch_outputs.append(pred)
                batch_features.append(wsi_feature.unsqueeze(0))
            
            pred = torch.cat(batch_outputs, dim=0)
            wsi_feature = torch.cat(batch_features, dim=0)  # (batch_size, 512)
        else:
            # Single sample processing
            x = self.feature(x.squeeze(0))
            x = x.squeeze(0)
            
            # Compute attention scores
            a = self.attention_a(x)
            b = self.attention_b(x)
            A = a.mul(b)  # Gated attention
            A = self.attention_c(A)
            A = torch.transpose(A, -1, -2)  # [K, num_patches]
            A = F.softmax(A, dim=-1)  # Softmax over patches
            
            # Weighted aggregation
            x_weighted = x * A.reshape(-1, 1)
            wsi_feature = x_weighted.sum(0)  # (512,) - aggregated feature
            pred = self.classifier(wsi_feature.unsqueeze(0))
        
        # Compute probabilities and predictions
        Y_prob = F.softmax(pred, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]
        result_dict = {}
        
        return pred, Y_prob, Y_hat, wsi_feature, result_dict 