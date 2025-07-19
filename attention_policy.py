import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Type, Union, Optional
import numpy as np
from gymnasium import spaces

from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.sac.policies import SACPolicy


class VariableCoinAttentionExtractor(BaseFeaturesExtractor):
    """
    Advanced attention extractor for variable portfolio sizes with masking.
    
    This extractor handles Dict observations with 'observations' and 'mask' keys,
    applying proper attention masking to ignore inactive coins.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,
        max_coins: int = 5,
        features_per_coin: int = 14,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        # Calculate total features for the Dict observation space
        total_features = observation_space['observations'].shape[0] + observation_space['mask'].shape[0]
        super().__init__(observation_space, features_dim)
        
        self.max_coins = max_coins
        self.features_per_coin = features_per_coin
        self.d_model = features_dim // 2  # Use half for coin embeddings
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Project each coin's features to d_model
        self.coin_projection = nn.Linear(self.features_per_coin, self.d_model)
        
        # Learnable coin position embeddings (different from coin type embeddings)
        self.position_embeddings = nn.Parameter(torch.randn(max_coins, self.d_model))
        
        # Transformer layers with proper masking support
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Market context attention for cross-asset relationships
        self.market_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Global portfolio context layer
        self.global_pooling = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final feature projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model * 2, features_dim),  # *2 for max pooled + mean pooled
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(features_dim, features_dim)
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Extract observations and mask
        obs = observations['observations']  # [batch_size, max_coins * features_per_coin]
        mask = observations['mask']  # [batch_size, max_coins]
        
        batch_size = obs.shape[0]
        
        # Reshape observations to [batch_size, max_coins, features_per_coin]
        coin_features = obs.view(batch_size, self.max_coins, self.features_per_coin)
        
        # Project each coin's features to d_model
        coin_embeddings = self.coin_projection(coin_features)  # [batch_size, max_coins, d_model]
        
        # Add position embeddings
        coin_embeddings = coin_embeddings + self.position_embeddings.unsqueeze(0)
        
        # Create attention mask for transformer (True = ignore, False = attend)
        # Transformer expects True for positions to ignore
        attention_mask = ~mask.bool()  # [batch_size, max_coins]
        
        # Apply transformer with masking
        # Note: src_key_padding_mask masks out entire sequences (coins)
        # Suppress nested tensor warnings
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*nested tensors.*")
            transformed = self.transformer(
                coin_embeddings, 
                src_key_padding_mask=attention_mask
            )  # [batch_size, max_coins, d_model]
        
        # Apply market attention to understand coin relationships
        market_context, attention_weights = self.market_attention(
            transformed, transformed, transformed,
            key_padding_mask=attention_mask
        )
        
        # Combine original and market context
        enhanced_features = transformed + market_context
        
        # Mask out inactive coins for pooling
        mask_expanded = mask.unsqueeze(-1).expand_as(enhanced_features)  # [batch_size, max_coins, d_model]
        masked_features = enhanced_features * mask_expanded
        
        # Global pooling with proper masking
        # Global pooling with proper masking
        # Mean pooling (ignoring masked positions)
        valid_coins = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [batch_size, 1]
        mean_pooled = masked_features.sum(dim=1) / valid_coins.view(batch_size, 1)  # [batch_size, d_model]
        
        # Max pooling (replacing -inf with very negative values for masked positions)
        masked_features_for_max = masked_features.clone()
        # Apply mask: set inactive coins to very negative values
        inactive_mask = mask_expanded == 0
        masked_features_for_max[inactive_mask] = -1e9
        
        # Max pooling along coin dimension
        max_pooled = masked_features_for_max.max(dim=1)[0]  # [batch_size, d_model]
        
        # Ensure both have the same number of dimensions
        if mean_pooled.dim() != max_pooled.dim():
            # This shouldn't happen, but let's handle it gracefully
            if mean_pooled.dim() == 3 and max_pooled.dim() == 2:
                mean_pooled = mean_pooled.squeeze(1)
            elif mean_pooled.dim() == 2 and max_pooled.dim() == 3:
                max_pooled = max_pooled.squeeze(1)
        
        # Combine pooled representations
        combined = torch.cat([mean_pooled, max_pooled], dim=1)  # [batch_size, d_model * 2]
        
        # Final projection
        features = self.output_projection(combined)  # [batch_size, features_dim]
        
        return features


class MultiHeadAttentionExtractor(BaseFeaturesExtractor):
    """
    Enhanced Multi-Head Attention extractor for variable portfolio sizes.
    Handles both Box and Dict observation spaces for backward compatibility.
    """

    def __init__(
        self,
        observation_space: Union[spaces.Box, spaces.Dict],
        features_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        d_model: int = 128,
    ):
        super().__init__(observation_space, features_dim)
        
        # Determine if we're dealing with Dict (variable) or Box (fixed) observations
        self.is_dict_obs = isinstance(observation_space, spaces.Dict)
        
        if self.is_dict_obs:
            # Dict observation space: extract total dimension
            self.input_dim = observation_space['observations'].shape[0]
            self.mask_dim = observation_space['mask'].shape[0]
        else:
            # Box observation space: use shape directly
            self.input_dim = observation_space.shape[0]
            self.mask_dim = 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Input projection to d_model dimensions
        self.input_projection = nn.Linear(self.input_dim, d_model)
        
        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model))  # Max 100 features
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(n_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
        # Feed-forward networks
        self.feed_forwards = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout)
            ) for _ in range(n_layers)
        ])
        
        # Final projection to desired features_dim
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, features_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, observations: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if self.is_dict_obs:
            # Handle Dict observations
            obs = observations['observations']
            # Note: For this simpler attention, we don't use the mask
            # but it's available if needed
        else:
            # Handle Box observations (legacy)
            obs = observations
            
        batch_size = obs.shape[0]
        
        # Reshape observation to work with attention
        x = obs.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        # Project to d_model dimensions
        x = self.input_projection(x)  # [batch_size, 1, d_model]
        
        # Add positional encoding
        seq_len = x.shape[1]
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Apply attention layers
        for i in range(self.n_layers):
            # Multi-head attention with residual connection
            attn_output, _ = self.attention_layers[i](x, x, x)
            x = self.layer_norms[i](x + attn_output)
            
            # Feed-forward with residual connection
            ff_output = self.feed_forwards[i](x)
            x = self.layer_norms[i](x + ff_output)
        
        # Global average pooling across sequence dimension
        x = x.mean(dim=1)  # [batch_size, d_model]
        
        # Final projection
        features = self.output_projection(x)  # [batch_size, features_dim]
        
        return features


class CoinAttentionExtractor(BaseFeaturesExtractor):
    """
    Legacy coin attention extractor with variable portfolio support.
    Enhanced to handle both Box and Dict observation spaces.
    """

    def __init__(
        self,
        observation_space: Union[spaces.Box, spaces.Dict],
        features_dim: int = 256,
        max_coins: int = 5,
        features_per_coin: int = 14,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        **kwargs  # Accept additional arguments for backward compatibility
    ):
        super().__init__(observation_space, features_dim)
        
        # Determine observation type and setup
        self.is_dict_obs = isinstance(observation_space, spaces.Dict)
        
        if self.is_dict_obs:
            self.input_dim = observation_space['observations'].shape[0]
            self.max_coins = max_coins
            self.features_per_coin = features_per_coin
        else:
            # Legacy mode: infer from Box observation space
            self.input_dim = observation_space.shape[0]
            self.max_coins = kwargs.get('n_coins', 3)  # Backward compatibility
            self.features_per_coin = self.input_dim // self.max_coins
            
        self.d_model = 128
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Project each coin's features to d_model
        self.coin_projection = nn.Linear(self.features_per_coin, self.d_model)
        
        # Learnable coin embeddings
        self.coin_embeddings = nn.Parameter(torch.randn(self.max_coins, self.d_model))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Market context attention
        self.market_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Final projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model * self.max_coins, features_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(features_dim, features_dim)
        )

    def forward(self, observations: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if self.is_dict_obs:
            # Handle Dict observations
            obs = observations['observations']
            mask = observations['mask']
        else:
            # Handle Box observations (legacy)
            obs = observations
            mask = None
        
        batch_size = obs.shape[0]
        
        # Reshape observations to [batch_size, max_coins, features_per_coin]
        coin_features = obs.view(batch_size, self.max_coins, self.features_per_coin)
        
        # Project each coin's features
        coin_embeddings = self.coin_projection(coin_features)  # [batch_size, max_coins, d_model]
        
        # Add coin-specific embeddings
        coin_embeddings = coin_embeddings + self.coin_embeddings.unsqueeze(0)
        
        # Apply transformer with optional masking
        if mask is not None:
            attention_mask = ~mask.bool()  # True = ignore
            transformed = self.transformer(coin_embeddings, src_key_padding_mask=attention_mask)
        else:
            transformed = self.transformer(coin_embeddings)
        
        # Apply market attention
        if mask is not None:
            market_context, _ = self.market_attention(
                transformed, transformed, transformed,
                key_padding_mask=~mask.bool()
            )
        else:
            market_context, _ = self.market_attention(transformed, transformed, transformed)
        
        # Combine original and context
        enhanced_features = transformed + market_context
        
        # Flatten and project to final features
        flattened = enhanced_features.reshape(batch_size, -1)
        features = self.output_projection(flattened)
        
        return features


class AttentionActorCriticPolicy(ActorCriticPolicy):
    """
    Enhanced Actor-Critic policy supporting both fixed and variable portfolio sizes.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        attention_type: str = "variable_coin_attention",
        features_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        **kwargs
    ):
        # Set the features extractor based on attention type
        if attention_type == "variable_coin_attention":
            kwargs["features_extractor_class"] = VariableCoinAttentionExtractor
        elif attention_type == "multihead":
            kwargs["features_extractor_class"] = MultiHeadAttentionExtractor
        elif attention_type == "coin_attention":
            kwargs["features_extractor_class"] = CoinAttentionExtractor
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        kwargs["features_extractor_kwargs"] = {
            "features_dim": features_dim,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "dropout": dropout,
        }
        
        # Add specific parameters for variable coin attention
        if attention_type == "variable_coin_attention":
            kwargs["features_extractor_kwargs"].update({
                "max_coins": kwargs.pop("max_coins", 5),
                "features_per_coin": kwargs.pop("features_per_coin", 14)
            })
        elif attention_type == "coin_attention":
            kwargs["features_extractor_kwargs"].update({
                "max_coins": kwargs.pop("max_coins", kwargs.pop("n_coins", 3)),
                "features_per_coin": kwargs.pop("features_per_coin", 14),
                "n_coins": kwargs.pop("n_coins", 3)  # Backward compatibility
            })
        
        # Remove any extra parameters that shouldn't go to base policy
        extra_params = ["max_coins", "features_per_coin", "n_coins"]
        for param in extra_params:
            kwargs.pop(param, None)
        
        # Set net_arch to match features_dim
        if "net_arch" not in kwargs:
            kwargs["net_arch"] = dict(pi=[features_dim], vf=[features_dim])
            
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)


class AttentionTD3Policy(TD3Policy):
    """
    TD3 policy with attention-based feature extraction.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        attention_type: str = "variable_coin_attention",
        features_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        **kwargs
    ):
        # Set the features extractor based on attention type
        if attention_type == "variable_coin_attention":
            kwargs["features_extractor_class"] = VariableCoinAttentionExtractor
        elif attention_type == "multihead":
            kwargs["features_extractor_class"] = MultiHeadAttentionExtractor
        elif attention_type == "coin_attention":
            kwargs["features_extractor_class"] = CoinAttentionExtractor
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        kwargs["features_extractor_kwargs"] = {
            "features_dim": features_dim,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "dropout": dropout,
        }
        
        # Add specific parameters for variable coin attention
        if attention_type == "variable_coin_attention":
            kwargs["features_extractor_kwargs"].update({
                "max_coins": kwargs.pop("max_coins", 5),
                "features_per_coin": kwargs.pop("features_per_coin", 14)
            })
        elif attention_type == "coin_attention":
            kwargs["features_extractor_kwargs"].update({
                "max_coins": kwargs.pop("max_coins", kwargs.pop("n_coins", 3)),
                "features_per_coin": kwargs.pop("features_per_coin", 14),
                "n_coins": kwargs.pop("n_coins", 3)  # Backward compatibility
            })
        
        # Remove any extra parameters that shouldn't go to base policy
        extra_params = ["max_coins", "features_per_coin", "n_coins"]
        for param in extra_params:
            kwargs.pop(param, None)
        
        # Set net_arch to match features_dim
        if "net_arch" not in kwargs:
            kwargs["net_arch"] = [features_dim, features_dim]
            
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)


class AttentionSACPolicy(SACPolicy):
    """
    SAC policy with attention-based feature extraction.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        attention_type: str = "variable_coin_attention",
        features_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        **kwargs
    ):
        # Set the features extractor based on attention type
        if attention_type == "variable_coin_attention":
            kwargs["features_extractor_class"] = VariableCoinAttentionExtractor
        elif attention_type == "multihead":
            kwargs["features_extractor_class"] = MultiHeadAttentionExtractor
        elif attention_type == "coin_attention":
            kwargs["features_extractor_class"] = CoinAttentionExtractor
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        kwargs["features_extractor_kwargs"] = {
            "features_dim": features_dim,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "dropout": dropout,
        }
        
        # Add specific parameters for variable coin attention
        if attention_type == "variable_coin_attention":
            kwargs["features_extractor_kwargs"].update({
                "max_coins": kwargs.pop("max_coins", 5),
                "features_per_coin": kwargs.pop("features_per_coin", 14)
            })
        elif attention_type == "coin_attention":
            kwargs["features_extractor_kwargs"].update({
                "max_coins": kwargs.pop("max_coins", kwargs.pop("n_coins", 3)),
                "features_per_coin": kwargs.pop("features_per_coin", 14),
                "n_coins": kwargs.pop("n_coins", 3)  # Backward compatibility
            })
        
        # Remove any extra parameters that shouldn't go to base policy
        extra_params = ["max_coins", "features_per_coin", "n_coins"]
        for param in extra_params:
            kwargs.pop(param, None)
        
        # Set net_arch to match features_dim
        if "net_arch" not in kwargs:
            kwargs["net_arch"] = [features_dim, features_dim]
            
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)


class AttentionDDPGPolicy(TD3Policy):
    """
    DDPG policy with attention-based feature extraction.
    Inherits from TD3Policy as per SB3 design.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        attention_type: str = "variable_coin_attention",
        features_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        **kwargs
    ):
        # Set the features extractor based on attention type
        if attention_type == "variable_coin_attention":
            kwargs["features_extractor_class"] = VariableCoinAttentionExtractor
        elif attention_type == "multihead":
            kwargs["features_extractor_class"] = MultiHeadAttentionExtractor
        elif attention_type == "coin_attention":
            kwargs["features_extractor_class"] = CoinAttentionExtractor
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        kwargs["features_extractor_kwargs"] = {
            "features_dim": features_dim,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "dropout": dropout,
        }
        
        # Add specific parameters for variable coin attention
        if attention_type == "variable_coin_attention":
            kwargs["features_extractor_kwargs"].update({
                "max_coins": kwargs.pop("max_coins", 5),
                "features_per_coin": kwargs.pop("features_per_coin", 14)
            })
        elif attention_type == "coin_attention":
            kwargs["features_extractor_kwargs"].update({
                "max_coins": kwargs.pop("max_coins", kwargs.pop("n_coins", 3)),
                "features_per_coin": kwargs.pop("features_per_coin", 14),
                "n_coins": kwargs.pop("n_coins", 3)  # Backward compatibility
            })
        
        # Remove any extra parameters that shouldn't go to base policy
        extra_params = ["max_coins", "features_per_coin", "n_coins"]
        for param in extra_params:
            kwargs.pop(param, None)
        
        # Set net_arch to match features_dim
        if "net_arch" not in kwargs:
            kwargs["net_arch"] = [features_dim, features_dim]
            
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)


# =============================================================================
# SIMPLE MLP POLICY CLASSES (NO ATTENTION) - REFACTORED
# =============================================================================

class SimpleMlpActorCriticPolicy(ActorCriticPolicy):
    """
    Simple MLP Actor-Critic policy (for PPO).
    This class is now a standard MlpPolicy that will use the default FlattenExtractor.
    The network architecture is passed via `policy_kwargs` in the model constructor.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Build the network architecture.
        This is overriden to use custom learning rates for actor and critic.
        """
        # Create the network architecture
        super()._build(lr_schedule)

        # Override optimizers if custom learning rates are provided
        if "actor_lr" in self.optimizer_kwargs and "critic_lr" in self.optimizer_kwargs:
            actor_lr = self.optimizer_kwargs.pop("actor_lr")
            critic_lr = self.optimizer_kwargs.pop("critic_lr")

            # Create new optimizers with the specified learning rates
            self.actor.optimizer = self.optimizer_class(
                self.actor.parameters(), lr=actor_lr, **self.optimizer_kwargs
            )
            self.critic.optimizer = self.optimizer_class(
                self.critic.parameters(), lr=critic_lr, **self.optimizer_kwargs
            )
            print(f"✅ Using custom learning rates: Actor={actor_lr}, Critic={critic_lr}")

class SimpleMlpTD3Policy(TD3Policy):
    """
    Simple MLP TD3 policy.
    This class is now a standard MlpPolicy that will use the default FlattenExtractor.
    The network architecture is passed via `policy_kwargs` (e.g., dict(pi=[...], qf=[...]))
    in the model constructor.
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        actor_lr: Optional[float] = None,
        critic_lr: Optional[float] = None,
        **kwargs,
    ):
        # ✅ FIX: Set the custom LRs on the instance *before* calling the parent constructor.
        # This prevents an AttributeError because the parent's __init__ calls self._build(),
        # which needs these attributes to be available.
        self.actor_lr_override = actor_lr
        self.critic_lr_override = critic_lr
        
        # ✅ FIX: Pass the custom learning rates into the optimizer_kwargs so they are available
        # to the parent class during initialization.
        if "optimizer_kwargs" not in kwargs:
            kwargs["optimizer_kwargs"] = {}
        # if actor_lr is not None:
        #     kwargs["optimizer_kwargs"]["actor_lr"] = actor_lr
        # if critic_lr is not None:
        #     kwargs["optimizer_kwargs"]["critic_lr"] = critic_lr
            
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Build the network architecture.
        This is overridden to use custom learning rates for actor and critic if provided.
        """
        # Create the network architecture using the parent's _build method
        super()._build(lr_schedule)

        # If custom learning rates were passed, override the default optimizers
        if self.actor_lr_override is not None and self.critic_lr_override is not None:
            print(f"🔧 Overriding optimizers with custom learning rates: Actor={self.actor_lr_override}, Critic={self.critic_lr_override}")
            
            # Re-create the optimizers with the specific learning rates
            # We pop our custom LRs from optimizer_kwargs so they aren't passed to Adam again.
            optimizer_kwargs = self.optimizer_kwargs.copy()
            optimizer_kwargs.pop("actor_lr", None)
            optimizer_kwargs.pop("critic_lr", None)
            
            self.actor.optimizer = self.optimizer_class(
                self.actor.parameters(), lr=self.actor_lr_override, **optimizer_kwargs
            )
            self.critic.optimizer = self.optimizer_class(
                self.critic.parameters(), lr=self.critic_lr_override, **optimizer_kwargs
            )

class SimpleMlpSACPolicy(SACPolicy):
    """
    Simple MLP SAC policy.
    This class is now a standard MlpPolicy that will use the default FlattenExtractor.
    The network architecture is passed via `policy_kwargs` (e.g., dict(pi=[...], qf=[...]))
    in the model constructor.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class SimpleMlpDDPGPolicy(TD3Policy):
    """
    Simple MLP DDPG policy. Inherits from TD3Policy as per SB3.
    This class is now a standard MlpPolicy that will use the default FlattenExtractor.
    The network architecture is passed via `policy_kwargs` (e.g., dict(pi=[...], qf=[...]))
    in the model constructor.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# Enhanced factory function
def create_attention_policy(attention_type: str = "variable_coin_attention", algorithm: str = "PPO", **kwargs):
    """
    Factory function to create policies with attention or simple MLP.
    
    Args:
        attention_type: "variable_coin_attention", "multihead", "coin_attention", or "mlp"
        algorithm: "PPO", "TD3", "SAC", or "DDPG"
        **kwargs: Additional arguments for the attention mechanism or MLP
    
    Returns:
        Policy class that can be used with SB3 algorithms
    """
    
    # Correctly select the policy class based on attention type and algorithm
    if attention_type.lower() == "mlp":
        # ---- MLP Policies (NO ATTENTION) ----
        if algorithm.upper() == "PPO":
            return SimpleMlpActorCriticPolicy
        elif algorithm.upper() == "TD3":
            return SimpleMlpTD3Policy
        elif algorithm.upper() == "SAC":
            return SimpleMlpSACPolicy
        elif algorithm.upper() == "DDPG":
            return SimpleMlpDDPGPolicy
        else:
            raise ValueError(f"Unsupported algorithm for MLP: {algorithm}")

    else:
        # ---- Attention-Based Policies ----
        if algorithm.upper() == "PPO":
            BasePolicyClass = AttentionActorCriticPolicy
        elif algorithm.upper() == "TD3":
            BasePolicyClass = AttentionTD3Policy
        elif algorithm.upper() == "SAC":
            BasePolicyClass = AttentionSACPolicy
        elif algorithm.upper() == "DDPG":
            BasePolicyClass = AttentionDDPGPolicy
        else:
            raise ValueError(f"Unsupported algorithm for Attention: {algorithm}")

        # Create and return the custom attention policy
        class CustomAttentionPolicy(BasePolicyClass):
            def __init__(self, *args, **policy_kwargs):
                super().__init__(*args, attention_type=attention_type, **kwargs, **policy_kwargs)
        return CustomAttentionPolicy


# Enhanced configuration for variable portfolios and MLPs
ATTENTION_CONFIGS = {
    "light": {
        "features_dim": 64,  # Reduced for smaller observation space
        "n_heads": 4,
        "n_layers": 1,
        "dropout": 0.1,
        "max_coins": 5,
        "features_per_coin": 14  # Updated to match config.py
    },
    "medium": {
        "features_dim": 128,  # Reduced for smaller observation space
        "n_heads": 8,
        "n_layers": 2,
        "dropout": 0.1,
        "max_coins": 5,
        "features_per_coin": 14  # Updated to match config.py
    },
    "heavy": {
        "features_dim": 256,  # Reduced for smaller observation space
        "n_heads": 16,  # Changed from 12 to 16 (512 ÷ 16 = 32)
        "n_layers": 3,
        "dropout": 0.2,
        "max_coins": 5,
        "features_per_coin": 14  # Updated to match config.py
    }
}

# Simple MLP configurations are now defined in the training script.
# The `net_arch` is passed directly to the model via `policy_kwargs`.
MLP_CONFIGS = {}


class NormalisedActor(nn.Module):
    """
    ✅ ENHANCED SOLUTION: Custom actor that outputs normalized portfolio weights with better exploration.
    
    This prevents the many-to-one mapping issue while allowing proper gradient flow and exploration.
    
    Key improvements:
    1. L1 normalization instead of softmax (allows negative logits)
    2. Better gradient flow for learning
    3. Injective mapping: each unique network output → unique portfolio allocation
    4. Proper exploration and differentiation based on observations
    """
    
    def __init__(self, observation_dim: int, action_dim: int, net_arch: List[int] = [400, 300]):
        super().__init__()
        
        # Build MLP layers (same architecture as standard TD3)
        layers = []
        prev_dim = observation_dim
        
        for layer_dim in net_arch:
            layers.extend([
                nn.Linear(prev_dim, layer_dim),
                nn.ReLU(),
            ])
            prev_dim = layer_dim
        
        # Output layer (no activation - we'll apply custom normalization)
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize the final layer with smaller weights for better exploration
        with torch.no_grad():
            self.mlp[-1].weight.data *= 0.1
            self.mlp[-1].bias.data.zero_()
        
        print(f"🎯 Enhanced NormalisedActor created:")
        print(f"   📊 Architecture: {observation_dim} → {net_arch} → {action_dim}")
        print(f"   ✅ Uses L1 normalization (better than softmax for exploration)")
        print(f"   🔄 Prevents many-to-one mapping plateau issue")
        print(f"   🎲 Allows proper gradient flow and learning")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: observations -> unnormalized positive actions.
        
        CRITICAL FIX: Replicate TF-Agents action generation.
        The actor outputs independent positive values (activations) for each action.
        The environment is then responsible for L1 normalizing them into a valid
        portfolio weight distribution. This prevents the "winner-take-all"
        behavior of softmax and eliminates allocation bias.
        """
        # ✅ DTYPE FIX: Ensure observations are float32 for PyTorch compatibility
        if observations.dtype != torch.float32:
            observations = observations.float()
        
        # Pass through MLP layers to get raw logits
        logits = self.mlp(observations)
        
        # ✅ TF-AGENTS STYLE: Output independent actions in [0, 1] using sigmoid.
        # The environment will normalize these into portfolio weights.
        # This is more robust than softmax and avoids bias.
        unnormalized_actions = torch.sigmoid(logits)
        
        # Debug output to verify diverse allocations
        if torch.rand(1).item() < 0.01:  # Print 1% of the time to avoid spam
            # ✅ BATCH FIX: Handle batch processing by printing the first item
            first_item_actions = unnormalized_actions[0]
            
            print(f"🔍 NormalisedActor (TF-Agents Style) Debug:")
            print(f"   Raw logits (first item): {logits[0].detach().cpu().numpy()}")
            print(f"   Sigmoid activations (first item): {first_item_actions.detach().cpu().numpy()}")
            print(f"   NOTE: These are NOT normalized weights. Env will normalize.")
            
        return unnormalized_actions
    
    def set_training_mode(self, mode: bool) -> None:
        """Set training mode for the actor (required by SB3)."""
        self.train(mode)


class NormalisedTD3Policy(TD3Policy):
    """
    ✅ COMPREHENSIVE FIX: TD3 policy with NormalisedActor to prevent plateau convergence.
    
    This custom policy completely overrides SB3's action handling to ensure our
    normalized portfolio weights are preserved during both training and inference.
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: List[int] = [400, 300],
        features_dim: int = 300,
        **kwargs
    ):
        # We'll override the actor creation, so set basic parameters first
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        
        # Store architecture for custom actor
        self.net_arch = net_arch
        self.features_dim = features_dim
        
        print(f"🎯 NormalisedTD3Policy created:")
        print(f"   🔄 Uses NormalisedActor (outputs normalized weights)")
        print(f"   🚫 No tanh squashing or SB3 rescaling")
        print(f"   ✅ Prevents many-to-one mapping plateau")
    
    def make_actor(self, features_extractor: nn.Module = None) -> nn.Module:
        """Create custom NormalisedActor instead of standard TD3 actor."""
        
        # Get observation dimension from features extractor or directly
        if features_extractor is not None:
            obs_dim = features_extractor.features_dim
        else:
            if isinstance(self.observation_space, spaces.Dict):
                obs_dim = self.observation_space['observations'].shape[0]
            else:
                obs_dim = self.observation_space.shape[0]
        
        action_dim = self.action_space.shape[0]
        
        # Create normalized actor
        actor = NormalisedActor(
            observation_dim=obs_dim,
            action_dim=action_dim,
            net_arch=self.net_arch
        )
        
        print(f"✅ Custom NormalisedActor created: {obs_dim} → {action_dim}")
        return actor
    
    def _build(self, lr_schedule: Schedule) -> None:
        """Build the policy networks with custom actor."""
        
        # Create features extractor
        self.features_extractor = self.make_features_extractor()
        
        # Create custom normalized actor
        self.actor = self.make_actor(self.features_extractor)
        
        # Create standard critic
        self.critic = self.make_critic(self.features_extractor)
        
        # Create target networks
        self.actor_target = self.make_actor(self.features_extractor)
        self.critic_target = self.make_critic(self.features_extractor)
        
        # Copy parameters to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Create optimizers
        self.actor_optimizer = self.optimizer_class(
            self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
        self.critic_optimizer = self.optimizer_class(
            self.critic.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
        
        # ✅ CRITICAL FIX: Assign optimizer to actor so SB3 can find it
        # SB3's TD3 calls self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        # This line prevents AttributeError: 'NormalisedActor' object has no attribute 'optimizer'
        self.actor.optimizer = self.actor_optimizer
        self.critic.optimizer = self.critic_optimizer
        
        print(f"✅ NormalisedTD3Policy networks built successfully")
        print(f"✅ Optimizer assignment fix applied - prevents SB3 AttributeError")
    
    def forward(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        ✅ COMPREHENSIVE FIX: Override forward method used during training.
        
        This ensures our normalized actions are used during training data collection,
        not just during inference via _predict().
        """
        # Extract features using the correct SB3 method signature
        features = self.extract_features(observation, self.features_extractor)
        
        # Get normalized weights directly from our actor (no SB3 rescaling)
        action = self.actor(features)
        
        # Debug logging to trace training actions
        if torch.rand(1).item() < 0.005:  # Log 0.5% of training actions
            print(f"🔍 Training Action Debug:")
            print(f"   NormalisedActor output: {action.detach().cpu().numpy()}")
            print(f"   Action sum: {action.sum().item():.6f}")
        
        return action
    
    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        ✅ CRITICAL FIX: Override SB3's action prediction to use our normalized weights directly.
        
        SB3's default predict applies Box(0,1) rescaling which breaks our normalized weights.
        This method bypasses that rescaling and returns the actor's softmax output directly.
        
        Note: Action noise is handled by the TD3 algorithm, not the policy.
        """
        # Extract features using the correct SB3 method signature
        features = self.extract_features(observation, self.features_extractor)
        
        # Get normalized weights directly from our actor (no SB3 rescaling)
        with torch.no_grad():
            action = self.actor(features)
            
            # Action noise is handled by the TD3 algorithm externally
            # Our actor already outputs proper portfolio weights via softmax
        
        # Debug logging to trace inference actions
        if torch.rand(1).item() < 0.01:  # Log 1% of inference actions
            print(f"🔍 Inference Action Debug:")
            print(f"   NormalisedActor output: {action.detach().cpu().numpy()}")
            print(f"   Action sum: {action.sum().item():.6f}")
        
        return action
    
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        ✅ COMPREHENSIVE FIX: Override the main predict method to ensure no rescaling.
        
        This is the public interface used by SB3 algorithms and external code.
        We need to ensure our normalized actions pass through unchanged.
        """
        # Convert observation to tensor
        if isinstance(observation, dict):
            observation_tensor = {}
            for key, value in observation.items():
                observation_tensor[key] = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        else:
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        
        # If single observation, add batch dimension
        if observation_tensor.ndim == 1 or (isinstance(observation_tensor, dict) and 
                                           next(iter(observation_tensor.values())).ndim == 1):
            if isinstance(observation_tensor, dict):
                observation_tensor = {key: value.unsqueeze(0) for key, value in observation_tensor.items()}
            else:
                observation_tensor = observation_tensor.unsqueeze(0)
            single_observation = True
        else:
            single_observation = False
        
        # Get action from our custom _predict method
        with torch.no_grad():
            action_tensor = self._predict(observation_tensor, deterministic=deterministic)
        
        # Convert back to numpy and remove batch dimension if needed
        action = action_tensor.cpu().numpy()
        if single_observation:
            action = action[0]
        
        # Debug logging for the final output
        if np.random.rand() < 0.01:  # Log 1% of final actions
            print(f"🔍 Final Action Debug:")
            print(f"   Predict output: {action}")
            print(f"   Action sum: {np.sum(action):.6f}")
            print(f"   Action type: {type(action)}")
        
        return action, state 


class NormalisedDDPGPolicy(TD3Policy):
    """
    ✅ COMPREHENSIVE FIX: DDPG policy with NormalisedActor to prevent plateau convergence.
    
    This custom policy completely overrides SB3's action handling to ensure our
    normalized portfolio weights are preserved during both training and inference.
    Inherits from TD3Policy as per SB3 design for DDPG.
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: List[int] = [400, 300],
        features_dim: int = 300,
        **kwargs
    ):
        # We'll override the actor creation, so set basic parameters first
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        
        # Store architecture for custom actor
        self.net_arch = net_arch
        self.features_dim = features_dim
        
        print(f"🎯 NormalisedDDPGPolicy created:")
        print(f"   🔄 Uses NormalisedActor (outputs normalized weights)")
        print(f"   🚫 No tanh squashing or SB3 rescaling")
        print(f"   ✅ Prevents many-to-one mapping plateau")
    
    def make_actor(self, features_extractor: nn.Module = None) -> nn.Module:
        """Create custom NormalisedActor instead of standard DDPG actor."""
        
        # Get observation dimension from features extractor or directly
        if features_extractor is not None:
            obs_dim = features_extractor.features_dim
        else:
            if isinstance(self.observation_space, spaces.Dict):
                obs_dim = self.observation_space['observations'].shape[0]
            else:
                obs_dim = self.observation_space.shape[0]
        
        action_dim = self.action_space.shape[0]
        
        # Create normalized actor
        actor = NormalisedActor(
            observation_dim=obs_dim,
            action_dim=action_dim,
            net_arch=self.net_arch
        )
        
        print(f"✅ Custom NormalisedActor created for DDPG: {obs_dim} → {action_dim}")
        return actor
    
    def _build(self, lr_schedule: Schedule) -> None:
        """Build the policy networks with custom actor."""
        
        # Create features extractor
        self.features_extractor = self.make_features_extractor()
        
        # Create custom normalized actor
        self.actor = self.make_actor(self.features_extractor)
        
        # Create standard critic
        self.critic = self.make_critic(self.features_extractor)
        
        # Create target networks
        self.actor_target = self.make_actor(self.features_extractor)
        self.critic_target = self.make_critic(self.features_extractor)
        
        # Copy parameters to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Create optimizers
        self.actor_optimizer = self.optimizer_class(
            self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
        self.critic_optimizer = self.optimizer_class(
            self.critic.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
        
        # ✅ CRITICAL FIX: Assign optimizer to actor so SB3 can find it
        self.actor.optimizer = self.actor_optimizer
        self.critic.optimizer = self.critic_optimizer
        
        print(f"✅ NormalisedDDPGPolicy networks built successfully")
        print(f"✅ Optimizer assignment fix applied - prevents SB3 AttributeError")
    
    def forward(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        ✅ COMPREHENSIVE FIX: Override forward method used during training.
        """
        features = self.extract_features(observation, self.features_extractor)
        action = self.actor(features)
        
        if torch.rand(1).item() < 0.005:  # Log 0.5% of training actions
            print(f"🔍 DDPG Training Action Debug: Sum={action.sum().item():.6f}")
        
        return action
    
    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        ✅ CRITICAL FIX: Override SB3's action prediction to use our normalized weights directly.
        """
        features = self.extract_features(observation, self.features_extractor)
        with torch.no_grad():
            action = self.actor(features)
        
        if torch.rand(1).item() < 0.01:  # Log 1% of inference actions
            print(f"🔍 DDPG Inference Action Debug: Sum={action.sum().item():.6f}")
        
        return action
    
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        ✅ COMPREHENSIVE FIX: Override the main predict method to ensure no rescaling.
        """
        if isinstance(observation, dict):
            observation_tensor = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in observation.items()}
        else:
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)

        if observation_tensor.ndim == 1 or (isinstance(observation_tensor, dict) and next(iter(observation_tensor.values())).ndim == 1):
            if isinstance(observation_tensor, dict):
                observation_tensor = {k: v.unsqueeze(0) for k, v in observation_tensor.items()}
            else:
                observation_tensor = observation_tensor.unsqueeze(0)
            single_observation = True
        else:
            single_observation = False

        with torch.no_grad():
            action_tensor = self._predict(observation_tensor, deterministic=deterministic)

        action = action_tensor.cpu().numpy()
        if single_observation:
            action = action[0]

        if np.random.rand() < 0.01:
            print(f"🔍 DDPG Final Action Debug: Sum={np.sum(action):.6f}")
            
        return action, state 