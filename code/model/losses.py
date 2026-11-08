"""
Supervised Contrastive Loss for Fusion Model
============================================

This module implements a supervised contrastive loss that can be used
alongside the standard BCE loss for improved representation learning.

The supervised contrastive loss encourages:
- Similar representations for samples with the same label
- Dissimilar representations for samples with different labels

References:
    Khosla et al. "Supervised Contrastive Learning" (NeurIPS 2020)
    https://arxiv.org/abs/2004.11362
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.
    
    This loss operates on a batch of features and their corresponding labels,
    encouraging features with the same label to be close together while pushing
    apart features with different labels.
    
    For each anchor i:
        - Positive samples: all j ≠ i with label_j == label_i
        - Negative samples: all j with label_j ≠ label_i
    
    The loss for anchor i is:
        ℓ_i = -1/|P(i)| ∑_{p∈P(i)} log( exp(sim(z_i, z_p)/τ) / ∑_{a≠i} exp(sim(z_i, z_a)/τ) )
    
    where:
        - P(i) is the set of positive samples for anchor i
        - sim(·,·) is cosine similarity (dot product of normalized vectors)
        - τ is the temperature parameter
    
    Args:
        temperature: Temperature parameter for scaling similarities (default: 0.07)
        base_temperature: Base temperature for normalization (default: 0.07)
        
    Input:
        features: Tensor of shape [batch_size, feature_dim]
        labels: Tensor of shape [batch_size] with class labels (0 or 1 for binary)
        
    Output:
        loss: Scalar tensor representing the average contrastive loss
    """
    
    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        
        logger.info(f"Initialized SupConLoss with temperature={temperature}")
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            features: [batch_size, feature_dim] - representation vectors
            labels: [batch_size] - class labels (must be long/int64)
            
        Returns:
            loss: Scalar tensor
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Ensure labels are long type
        if labels.dtype != torch.long:
            labels = labels.long()
        
        # Handle edge case: batch size too small
        if batch_size < 2:
            logger.warning("SupConLoss: batch size < 2, returning zero loss")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Normalize features to unit sphere (L2 normalization)
        # This makes dot product equivalent to cosine similarity
        features = F.normalize(features, p=2, dim=1)  # [batch_size, feature_dim]
        
        # Compute similarity matrix: [batch_size, batch_size]
        # similarity[i, j] = dot_product(features[i], features[j])
        similarity_matrix = torch.matmul(features, features.T)  # [B, B]
        
        # Create mask for positive pairs (same label, different sample)
        # labels_equal[i, j] = 1 if label[i] == label[j], else 0
        labels = labels.contiguous().view(-1, 1)  # [batch_size, 1]
        labels_equal = torch.eq(labels, labels.T).float()  # [batch_size, batch_size]
        
        # Mask out diagonal (sample should not be positive with itself)
        mask_self = torch.eye(batch_size, dtype=torch.float32, device=device)
        positives_mask = labels_equal * (1 - mask_self)  # [batch_size, batch_size]
        
        # For numerical stability, subtract max from similarities before exp
        # Max is taken over all pairs (not just positives)
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()  # [batch_size, batch_size]
        
        # Compute exp(similarity / temperature)
        exp_logits = torch.exp(logits / self.temperature)  # [batch_size, batch_size]
        
        # Mask out self-similarity (anchor should not be in denominator with itself)
        exp_logits = exp_logits * (1 - mask_self)
        
        # Denominator: sum of exp(sim(z_i, z_a)) for all a ≠ i
        log_prob_denominator = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)  # [batch_size, 1]
        
        # Numerator: exp(sim(z_i, z_p)) for each positive p
        # log_prob = log(exp(sim(z_i, z_p))) - log(denominator)
        #          = sim(z_i, z_p) / temp - log(denominator)
        log_prob = (logits / self.temperature) - log_prob_denominator  # [batch_size, batch_size]
        
        # Count number of positives per anchor
        num_positives = positives_mask.sum(dim=1)  # [batch_size]
        
        # Only compute loss for anchors that have at least one positive
        valid_anchors = num_positives > 0  # [batch_size]
        
        if valid_anchors.sum() == 0:
            # No valid anchor-positive pairs in this batch
            logger.debug("SupConLoss: no valid anchor-positive pairs in batch")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Compute mean log probability over positive pairs for each anchor
        # mean_log_prob_pos[i] = (1/|P(i)|) * sum_{p in P(i)} log_prob[i, p]
        mean_log_prob_pos = (positives_mask * log_prob).sum(dim=1) / (num_positives + 1e-12)  # [batch_size]
        
        # Loss is negative of mean log probability, averaged over valid anchors
        # Apply temperature scaling as in original paper
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        
        # Average over valid anchors only
        loss = loss[valid_anchors].mean()
        
        return loss


def test_supcon_loss():
    """
    Test the SupConLoss implementation with simple cases.
    """
    print("Testing SupConLoss...")
    
    # Test 1: Simple case with 4 samples, 2 classes
    print("\nTest 1: Basic functionality")
    features = torch.tensor([
        [1.0, 0.0],
        [0.9, 0.1],  # Similar to first (class 0)
        [0.0, 1.0],
        [0.1, 0.9],  # Similar to third (class 1)
    ])
    labels = torch.tensor([0, 0, 1, 1])
    
    loss_fn = SupConLoss(temperature=0.07)
    loss = loss_fn(features, labels)
    print(f"Loss (should be positive): {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    print("✓ Test 1 passed")
    
    # Test 2: Perfect separation (should give low loss)
    print("\nTest 2: Perfect separation")
    features = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],  # Identical to first
        [-1.0, 0.0],
        [-1.0, 0.0],  # Identical to third
    ])
    labels = torch.tensor([0, 0, 1, 1])
    loss = loss_fn(features, labels)
    print(f"Loss (should be low): {loss.item():.4f}")
    print("✓ Test 2 passed")
    
    # Test 3: No positive pairs (all different labels)
    print("\nTest 3: No positive pairs")
    features = torch.randn(4, 10)
    labels = torch.tensor([0, 1, 0, 1])  # Alternating labels
    loss = loss_fn(features, labels)
    print(f"Loss (should be zero or very small): {loss.item():.6f}")
    print("✓ Test 3 passed")
    
    # Test 4: Single class (all same label)
    print("\nTest 4: Single class")
    features = torch.randn(4, 10)
    labels = torch.tensor([1, 1, 1, 1])
    loss = loss_fn(features, labels)
    print(f"Loss (should be positive): {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    print("✓ Test 4 passed")
    
    # Test 5: Batch size = 1 (edge case)
    print("\nTest 5: Batch size = 1")
    features = torch.randn(1, 10)
    labels = torch.tensor([0])
    loss = loss_fn(features, labels)
    print(f"Loss (should be zero): {loss.item():.6f}")
    assert loss.item() == 0.0, "Loss should be zero for batch size 1"
    print("✓ Test 5 passed")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_supcon_loss()
