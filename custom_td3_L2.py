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


class CustomTD3(TD3):
    """
    Custom TD3 Agent that uses Huber Loss for the critic update.
    This makes training more robust to outlier rewards.
    """

    def __init__(
        self,
        policy: Union[str, Type[TD3Policy]],
        env: Union[GymEnv, str],
        # ────── new hyperparameter ──────
        action_reg_coef: float = 0.1, # orginal0.1,
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
        self.max_grad_norm    = max_grad_norm
        self.action_reg_coef  = action_reg_coef

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

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """
        Override the default training loop to use Huber loss for the critic
        and apply gradient clipping.
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

            # with torch.no_grad():
            #     # Select action according to policy and add clipped noise
            #     noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
            #     noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            #     next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

            with torch.no_grad():
                # ✅ Correct: sample fresh Gaussian noise shaped by action_dim
                batch_size = replay_data.next_observations.shape[0]
                action_dim = self.action_space.shape[0] #action_dim = self.actor.action_dim  # Infer from actor
                device = self.device

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
            # ✅ GRADIENT CLIPPING: Clip the gradients to prevent explosions
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # OLD:
                # Compute actor loss
                # actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()


                # ─── Standard TD3 actor objective ───
                obs      = replay_data.observations
                next_obs = replay_data.next_observations

                a_cur  = self.actor(obs)
                a_next = self.actor(next_obs)

                # maximize Q(s,a) → minimize −Q
                actor_loss = -self.critic.q1_forward(obs, a_cur).mean()

                # ─── add temporal‐difference smoothness term ───
                # L2 regularizer on the action change between π(s_t) and π(s_{t+1})
                # penalize big jumps between π(s_t) and π(s_{t+1})
                smooth_loss = self.action_reg_coef * (
                    (a_next - a_cur)
                    .pow(2)
                    .sum(dim=1)
                    .mean()
                )
                
                actor_loss = actor_loss + smooth_loss
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                # ✅ GRADIENT CLIPPING: Clip the gradients for the actor as well
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor.optimizer.step()

                # Update target networks
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses)) 