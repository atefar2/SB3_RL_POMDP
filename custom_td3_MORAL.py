"""
Custom TD3 implementation for Stable Baselines 3 to introduce:
1. Huber Loss for the critic for more robust training.
2. (Reference) Support for periodic target updates via the `policy_delay` parameter.
"""

import numpy as np
import torch
from torch.nn import functional as F

from stable_baselines3 import TD3
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import polyak_update
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.td3.policies import TD3Policy

import gymnasium as gym

class LinearScalarizationWrapper(gym.Wrapper):
    def __init__(self, env, weights):
        super().__init__(env)
        self.weights = np.array(weights)
    
    def __getattr__(self, name):
        """Forward attribute access to the wrapped environment."""
        return getattr(self.env, name)

    def step(self, action):
        obs, reward_vector, terminated, truncated, info = self.env.step(action)
        
        # Handle both vectorized rewards and scalar rewards (e.g., at episode end)
        if hasattr(reward_vector, '__len__') and len(reward_vector) > 1:
            # Vectorized reward case
            scalar_reward = np.dot(reward_vector, self.weights)
            original_reward_vector = np.array(reward_vector)  # Ensure it's a numpy array
        else:
            # Scalar reward case (e.g., at episode termination)
            scalar_reward = float(reward_vector)
            original_reward_vector = np.array([scalar_reward, 0.0])  # Create dummy vector for consistency
        
        # Ensure scalar_reward is a Python float, not numpy array
        # Handle all possible numpy array forms robustly
        if np.isscalar(scalar_reward):
            scalar_reward = float(scalar_reward)
        elif hasattr(scalar_reward, 'size') and scalar_reward.size == 1:
            scalar_reward = float(scalar_reward.flatten()[0])  # Handle any 1-element array
        else:
            # If somehow we get a multi-element array, sum it (shouldn't happen with proper dot product)
            scalar_reward = float(np.sum(scalar_reward))
        
        # Update info to show the scalarized reward and components
        info['original_reward_vector'] = original_reward_vector.copy()  # Keep original for debugging
        info['scalarization_weights'] = self.weights.copy()
        info['reward'] = scalar_reward  # Override with scalar reward
        
        # Debug logging (occasionally)
        if np.random.rand() < 0.01:  # Log 1% of the time
            if len(original_reward_vector) > 1:
                print(f"🔄 LinearScalarizationWrapper: [{original_reward_vector[0]:.4f}, {original_reward_vector[1]:.4f}] -> {scalar_reward:.4f}")
            else:
                print(f"🔄 LinearScalarizationWrapper: {scalar_reward:.4f} (scalar)")
        
        return obs, scalar_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info


class CustomTD3(TD3):
    """
    Custom TD3 Agent that uses Huber Loss for the critic update.
    This makes training more robust to outlier rewards.
    """

    def __init__(
        self,
        policy: Union[str, Type[TD3Policy]],
        env: Union[GymEnv, str],
        # ────── removed action_reg_coef parameter ──────
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "step"),
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        # Add our custom arguments
        max_grad_norm: float = 1.0,
    ):
        # Store our custom args
        self.max_grad_norm = max_grad_norm
        # Removed: self.action_reg_coef = action_reg_coef

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            policy_delay=policy_delay,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
    """
    Custom TD3 Agent that uses Huber Loss for the critic update.
    This makes training more robust to outlier rewards.
    """

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """
        Override the default training loop to use Huber loss for the critic.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with torch.no_grad():
                batch_size = replay_data.next_observations.shape[0]
                action_dim = self.action_space.shape[0] #action_dim = self.actor.action_dim  # Infer from actor
                device = self.device                
                # Select action according to policy and add clipped noise
                noise = torch.normal(
                    mean=0.0,
                    std=self.target_policy_noise,
                    size=(batch_size, action_dim),
                    device=device
                ).clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the target Q value
                target_q_values = torch.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                target_q_values, _ = torch.min(target_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q_values

            # Get current Q-values estimates
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # ✅ HUBER LOSS: Compute critic loss using smooth_l1_loss (Huber loss)
            # This is less sensitive to outliers than the default MSE loss.
            critic_loss = sum(F.smooth_l1_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor.optimizer.step()

                # Update target networks
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses)) 