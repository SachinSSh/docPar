"""
Advanced KL Divergence Implementations for Machine Learning
============================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# ============================================================
# 1. VAE KL Divergence (Gaussian)
# ============================================================

def vae_kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL divergence between N(mu, var) and N(0, 1)
    
    Formula: KL = -0.5 * sum(1 + log(var) - mu^2 - var)
    
    Args:
        mu: Mean of latent distribution [batch_size, latent_dim]
        logvar: Log variance of latent distribution [batch_size, latent_dim]
    
    Returns:
        KL divergence per sample [batch_size]
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


class VAE(nn.Module):
    """Simple VAE with KL divergence loss"""
    
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar):
        """VAE loss = Reconstruction + KL"""
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        KLD = vae_kl_loss(mu, logvar).sum()
        return BCE + KLD


# ============================================================
# 2. Policy Gradient KL (Reinforcement Learning)
# ============================================================

def kl_categorical(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """
    KL divergence between two categorical distributions
    
    Args:
        p_logits: Logits of distribution P [batch_size, num_actions]
        q_logits: Logits of distribution Q [batch_size, num_actions]
    
    Returns:
        KL(P || Q) per sample
    """
    p = F.softmax(p_logits, dim=-1)
    log_p = F.log_softmax(p_logits, dim=-1)
    log_q = F.log_softmax(q_logits, dim=-1)
    
    return torch.sum(p * (log_p - log_q), dim=-1)


def kl_gaussian(mu1: torch.Tensor, logvar1: torch.Tensor, 
                mu2: torch.Tensor, logvar2: torch.Tensor) -> torch.Tensor:
    """
    KL divergence between two Gaussian distributions
    
    Formula: KL(P||Q) = 0.5 * [log(var2/var1) + (var1 + (mu1-mu2)^2)/var2 - 1]
    """
    var1 = logvar1.exp()
    var2 = logvar2.exp()
    
    kl = 0.5 * (logvar2 - logvar1 + (var1 + (mu1 - mu2).pow(2)) / var2 - 1)
    return kl.sum(dim=-1)


class PPOClipLoss(nn.Module):
    """PPO loss with KL divergence penalty"""
    
    def __init__(self, clip_epsilon=0.2, kl_coef=0.01):
        super().__init__()
        self.clip_epsilon = clip_epsilon
        self.kl_coef = kl_coef
    
    def forward(self, old_logits, new_logits, actions, advantages):
        """
        PPO clipped loss with KL penalty
        
        Args:
            old_logits: Policy logits from old policy
            new_logits: Policy logits from new policy
            actions: Actions taken
            advantages: Advantage estimates
        """
        # Probability ratios
        old_log_probs = F.log_softmax(old_logits, dim=-1)
        new_log_probs = F.log_softmax(new_logits, dim=-1)
        
        old_action_log_probs = old_log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        new_action_log_probs = new_log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        ratio = torch.exp(new_action_log_probs - old_action_log_probs)
        
        # Clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # KL penalty
        kl_div = kl_categorical(old_logits, new_logits).mean()
        
        return policy_loss + self.kl_coef * kl_div, kl_div


# ============================================================
# 3. Knowledge Distillation
# ============================================================

def distillation_loss(student_logits: torch.Tensor, 
                     teacher_logits: torch.Tensor,
                     temperature: float = 3.0,
                     alpha: float = 0.5) -> torch.Tensor:
    """
    Knowledge distillation using KL divergence
    
    Args:
        student_logits: Student model outputs [batch_size, num_classes]
        teacher_logits: Teacher model outputs [batch_size, num_classes]
        temperature: Softmax temperature (higher = softer)
        alpha: Weight for distillation loss
    
    Returns:
        Combined distillation loss
    """
    # Soft targets from teacher
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    
    # KL divergence (scaled by temperature^2)
    kl_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
    kl_loss = kl_loss * (temperature ** 2)
    
    return kl_loss


# ============================================================
# 4. Mutual Information Estimation
# ============================================================

def mutual_information_neural_estimation(x: torch.Tensor, 
                                        y: torch.Tensor,
                                        critic: nn.Module) -> torch.Tensor:
    """
    Estimate mutual information I(X; Y) using neural estimation
    Based on MINE (Mutual Information Neural Estimation)
    
    I(X;Y) = KL(P(X,Y) || P(X)P(Y))
    """
    # Joint distribution
    joint = torch.cat([x, y], dim=1)
    t_joint = critic(joint)
    
    # Marginal distribution (shuffle y)
    y_shuffle = y[torch.randperm(y.size(0))]
    marginal = torch.cat([x, y_shuffle], dim=1)
    t_marginal = critic(marginal)
    
    # MI lower bound: E[T(x,y)] - log(E[e^T(x,y')])
    mi_lower_bound = t_joint.mean() - torch.log(torch.exp(t_marginal).mean())
    
    return mi_lower_bound


# ============================================================
# 5. Importance Weighted Autoencoders (IWAE)
# ============================================================

def iwae_loss(x: torch.Tensor, 
             encoder: nn.Module, 
             decoder: nn.Module,
             num_samples: int = 5) -> torch.Tensor:
    """
    IWAE uses importance weighting for tighter bound
    
    log p(x) >= E[log(1/K * sum_k w_k)]
    where w_k = p(x|z_k)p(z_k) / q(z_k|x)
    """
    batch_size = x.size(0)
    
    # Encode
    mu, logvar = encoder(x)
    
    # Sample multiple z values
    mu = mu.unsqueeze(1).expand(-1, num_samples, -1)
    logvar = logvar.unsqueeze(1).expand(-1, num_samples, -1)
    
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + eps * std
    
    # Decode
    z_flat = z.view(-1, z.size(-1))
    recon_flat = decoder(z_flat)
    recon = recon_flat.view(batch_size, num_samples, -1)
    
    # Compute log weights
    x_expanded = x.unsqueeze(1).expand(-1, num_samples, -1)
    log_p_x_z = F.binary_cross_entropy(recon, x_expanded, reduction='none').sum(-1)
    
    log_p_z = -0.5 * (z.pow(2).sum(-1) + np.log(2 * np.pi) * z.size(-1))
    log_q_z_x = -0.5 * ((z - mu).pow(2) / std.pow(2) + logvar + np.log(2 * np.pi)).sum(-1)
    
    log_w = -log_p_x_z + log_p_z - log_q_z_x
    
    # Importance weighted bound
    return -torch.logsumexp(log_w, dim=1).mean() + np.log(num_samples)


# ============================================================
# 6. Batch Statistics
# ============================================================

def batch_kl_divergence(distributions_p: list, distributions_q: list) -> np.ndarray:
    """
    Compute KL divergence for batches of distributions
    
    Args:
        distributions_p: List of probability arrays
        distributions_q: List of probability arrays
    
    Returns:
        Array of KL divergences
    """
    kl_values = []
    for p, q in zip(distributions_p, distributions_q):
        p = np.asarray(p)
        q = np.asarray(q)
        kl = np.sum(np.where(p != 0, p * np.log(p / q), 0))
        kl_values.append(kl)
    return np.array(kl_values)


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Advanced KL Divergence Examples")
    print("=" * 60)
    
    # 1. VAE KL Loss
    print("\n1. VAE KL Divergence:")
    mu = torch.randn(32, 20)  # batch_size=32, latent_dim=20
    logvar = torch.randn(32, 20)
    kl = vae_kl_loss(mu, logvar)
    print(f"   Mean KL per sample: {kl.mean().item():.4f}")
    
    # 2. Categorical KL (for discrete actions)
    print("\n2. Categorical KL (Policy Gradient):")
    old_logits = torch.randn(64, 4)  # 64 samples, 4 actions
    new_logits = torch.randn(64, 4)
    kl = kl_categorical(old_logits, new_logits)
    print(f"   Mean KL: {kl.mean().item():.4f}")
    
    # 3. Gaussian KL
    print("\n3. Gaussian KL Divergence:")
    mu1, logvar1 = torch.randn(10, 5), torch.randn(10, 5)
    mu2, logvar2 = torch.randn(10, 5), torch.randn(10, 5)
    kl = kl_gaussian(mu1, logvar1, mu2, logvar2)
    print(f"   Mean KL: {kl.mean().item():.4f}")
    
    # 4. Knowledge Distillation
    print("\n4. Knowledge Distillation:")
    student_logits = torch.randn(32, 10)
    teacher_logits = torch.randn(32, 10)
    loss = distillation_loss(student_logits, teacher_logits)
    print(f"   Distillation loss: {loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
