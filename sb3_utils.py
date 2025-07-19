#!/usr/bin/env python3
"""
Stable Baselines 3 utility functions - SB3/Gymnasium equivalents to tf_agents functions.

This file provides SB3-compatible versions of common tf_agents functionality.
"""

import numpy as np
import torch
from typing import Optional, Union, Dict, Any
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv
import gymnasium as gym


# =============================================================================
# SB3/GYMNASIUM EQUIVALENTS TO TF_AGENTS FUNCTIONS
# =============================================================================

def compute_avg_return_sb3(
    model: BaseAlgorithm, 
    env: Union[gym.Env, VecEnv], 
    num_episodes: int = 10,
    deterministic: bool = True
) -> float:
    """
    SB3 equivalent to tf_agents compute_avg_return.
    
    Args:
        model: Trained SB3 model
        env: Gymnasium environment
        num_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic actions
    
    Returns:
        Average return across episodes
    """
    total_return = 0.0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_return = 0.0
        terminated = truncated = False
        
        while not (terminated or truncated):
            # SB3 way: model.predict() instead of policy.action()
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # Gymnasium way: step returns (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
        
        total_return += episode_return
    
    avg_return = total_return / num_episodes
    return float(avg_return)


def evaluate_policy_sb3(
    model: BaseAlgorithm,
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    return_episode_rewards: bool = False
) -> Union[float, tuple]:
    """
    Enhanced SB3 policy evaluation with detailed metrics.
    
    Args:
        model: Trained SB3 model
        env: Environment to evaluate on
        n_eval_episodes: Number of episodes
        deterministic: Use deterministic actions
        render: Whether to render episodes
        return_episode_rewards: Return individual episode rewards
    
    Returns:
        Average reward (and optionally episode rewards list)
    """
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_eval_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        terminated = truncated = False
        
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    print(f"📊 Evaluation Results:")
    print(f"   Mean reward: {mean_reward:.4f} ± {std_reward:.4f}")
    print(f"   Mean episode length: {mean_length:.1f}")
    print(f"   Min/Max rewards: {np.min(episode_rewards):.4f}/{np.max(episode_rewards):.4f}")
    
    if return_episode_rewards:
        return mean_reward, episode_rewards
    return mean_reward


def collect_rollouts_sb3(
    model: BaseAlgorithm,
    env: Union[gym.Env, VecEnv],
    n_steps: int,
    deterministic: bool = False
) -> Dict[str, np.ndarray]:
    """
    SB3 way to collect rollouts (equivalent to tf_agents collect_data).
    
    Args:
        model: SB3 model
        env: Environment
        n_steps: Number of steps to collect
        deterministic: Use deterministic actions
    
    Returns:
        Dictionary with collected data
    """
    observations = []
    actions = []
    rewards = []
    dones = []
    
    obs, _ = env.reset()
    
    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        dones.append(terminated or truncated)
        
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs
    
    return {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'dones': np.array(dones)
    }


def compute_portfolio_metrics(
    model: BaseAlgorithm,
    env: Union[gym.Env, VecEnv],
    n_episodes: int = 10
) -> Dict[str, float]:
    """
    Compute portfolio-specific evaluation metrics.
    
    Args:
        model: Trained SB3 model
        env: Portfolio environment
        n_episodes: Number of episodes to evaluate
    
    Returns:
        Dictionary of portfolio metrics
    """
    episode_returns = []
    portfolio_values = []
    portfolio_weights_history = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_return = 0.0
        terminated = truncated = False
        episode_values = []
        episode_weights = []
        
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_return += reward
            
            # Extract portfolio information from info dict
            if 'value' in info:
                episode_values.append(info['value'])
            if 'money_split' in info:
                episode_weights.append(info['money_split'])
        
        episode_returns.append(episode_return)
        if episode_values:
            portfolio_values.extend(episode_values)
        if episode_weights:
            portfolio_weights_history.extend(episode_weights)
    
    # Calculate metrics
    metrics = {
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'sharpe_ratio': np.mean(episode_returns) / (np.std(episode_returns) + 1e-8),
        'max_return': np.max(episode_returns),
        'min_return': np.min(episode_returns),
    }
    
    if portfolio_values:
        portfolio_values = np.array(portfolio_values)
        metrics.update({
            'final_portfolio_value': portfolio_values[-1],
            'max_drawdown': compute_max_drawdown(portfolio_values),
            'total_return': (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        })
    
    return metrics


def compute_max_drawdown(portfolio_values: np.ndarray) -> float:
    """Compute maximum drawdown from portfolio values."""
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    return np.min(drawdown)


# =============================================================================
# SB3 TRAINING UTILITIES
# =============================================================================

def train_with_periodic_evaluation(
    model: BaseAlgorithm,
    total_timesteps: int,
    eval_env: Union[gym.Env, VecEnv],
    eval_freq: int = 10000,
    n_eval_episodes: int = 10,
    verbose: bool = True
) -> Dict[str, list]:
    """
    Train model with periodic evaluation (SB3 way).
    
    Args:
        model: SB3 model to train
        total_timesteps: Total training steps
        eval_env: Environment for evaluation
        eval_freq: Frequency of evaluation
        n_eval_episodes: Episodes per evaluation
        verbose: Print progress
    
    Returns:
        Training history dictionary
    """
    history = {
        'timesteps': [],
        'eval_rewards': [],
        'eval_stds': []
    }
    
    for timestep in range(0, total_timesteps, eval_freq):
        # Train for eval_freq steps
        model.learn(total_timesteps=min(eval_freq, total_timesteps - timestep))
        
        # Evaluate
        mean_reward, episode_rewards = evaluate_policy_sb3(
            model, eval_env, n_eval_episodes, return_episode_rewards=True
        )
        
        history['timesteps'].append(timestep + eval_freq)
        history['eval_rewards'].append(mean_reward)
        history['eval_stds'].append(np.std(episode_rewards))
        
        if verbose:
            print(f"Timestep {timestep + eval_freq}: Mean reward = {mean_reward:.4f}")
    
    return history


# =============================================================================
# COMPARISON TABLE: TF_AGENTS VS SB3
# =============================================================================

def print_tf_agents_vs_sb3_comparison():
    """Print comparison between tf_agents and SB3 approaches."""
    
    comparison = """
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                     TF_AGENTS vs STABLE BASELINES 3                     ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║ FUNCTIONALITY          │ TF_AGENTS               │ STABLE BASELINES 3   ║
    ╠════════════════════════╪═════════════════════════╪══════════════════════╣
    ║ Environment Base       │ py_environment.PyEnv    │ gymnasium.Env        ║
    ║ Action/Obs Specs       │ array_spec.BoundedArray │ gym.spaces.Box       ║
    ║ Time Steps             │ ts.restart/transition   │ (obs, reward, term..)║
    ║ Policy Interface       │ policy.action(timestep) │ model.predict(obs)   ║
    ║ Training               │ agent.train(experience) │ model.learn(steps)   ║
    ║ Evaluation             │ compute_avg_return()    │ evaluate_policy()    ║
    ║ Data Collection        │ collect_step/collect_data│ Built into .learn()  ║
    ║ Replay Buffer          │ tf_uniform_replay_buffer │ Built into off-policy║
    ║ Trajectories           │ trajectory.from_trans.. │ Automatic in SB3     ║
    ║ Networks               │ Custom tf.keras nets    │ Custom torch policies║
    ║ Callbacks              │ Manual implementation   │ Built-in callbacks   ║
    ╚════════════════════════╧═════════════════════════╧══════════════════════╝
    
    ✅ SB3 ADVANTAGES:
    • Simpler API and less boilerplate code
    • Built-in evaluation, logging, and callbacks
    • Better documentation and community support
    • Automatic hyperparameter tuning support
    • Cleaner PyTorch-based implementation
    
    ❌ TF_AGENTS DISADVANTAGES:
    • More complex, requires manual implementation of many features
    • TensorFlow 1.x legacy issues
    • More verbose code for basic operations
    • Less active development and community
    """
    
    print(comparison)


if __name__ == "__main__":
    print("🔄 SB3 Utility Functions - TF_AGENTS to SB3 Migration")
    print_tf_agents_vs_sb3_comparison() 