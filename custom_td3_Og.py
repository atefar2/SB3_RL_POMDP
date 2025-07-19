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


class CustomTD3(TD3):
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