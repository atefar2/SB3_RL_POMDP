# %%
#!/usr/bin/env python3
"""
Evaluation script for RL portfolio allocation using Stable Baselines 3.
Compatible with attention-based policies and gymnasium environments.
"""

import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from typing import List, Dict, Any
from attention_policy import (
    SimpleMlpTD3Policy, 
    SimpleMlpSACPolicy, 
    SimpleMlpDDPGPolicy, 
    SimpleMlpActorCriticPolicy,
    NormalisedTD3Policy,
    NormalisedDDPGPolicy,
    AttentionTD3Policy,
    AttentionSACPolicy,
    AttentionDDPGPolicy,
    AttentionActorCriticPolicy,
    create_attention_policy
)
from custom_td3 import CustomTD3
import warnings
warnings.filterwarnings('ignore')

import config
from enviorment import PortfolioEnv


def evaluate_policy_sb3(model, env, num_episodes=1, deterministic=True, render=False):
    """
    Evaluate a Stable Baselines 3 policy and collect detailed episode data.
    Compatible with both fixed and variable portfolio environments.
    
    Args:
        model: Trained SB3 model
        env: Portfolio environment
        num_episodes: Number of episodes to run
        deterministic: Use deterministic actions
        render: Whether to render the environment
    
    Returns:
        List of episode data dictionaries
    """
    all_episode_data = []

    for episode in range(num_episodes):
        print(f"Running evaluation episode {episode + 1}/{num_episodes}...")
        
        # Reset environment
        obs, info = env.reset()
        
        # Detect portfolio type and extract episode coins
        is_variable_portfolio = hasattr(env, 'use_variable_portfolio') and env.use_variable_portfolio
        episode_coins = info.get('episode_coins', config.COINS) if is_variable_portfolio else config.COINS
        
        print(f"  Episode coins: {episode_coins} ({'variable' if is_variable_portfolio else 'fixed'} portfolio)")
        
        episode_data = {
            "timestamps": [],
            "values": [],
            "allocations": [],
            "market_data": [],
            "rewards": [],
            "actions": [],
            "observations": [],
            "episode_coins": episode_coins,  # Store episode-specific coins
            "is_variable_portfolio": is_variable_portfolio
        }
        
        # Store initial state
        episode_data["timestamps"].append(env.current_time)
        episode_data["values"].append(env.current_value)
        episode_data["allocations"].append(env.money_split_ratio.copy())
        episode_data["market_data"].append(env.dfslice.copy())
        episode_data["rewards"].append(0.0)  # Initial reward is 0
        
        # Handle initial action based on portfolio type
        if is_variable_portfolio:
            initial_action = [0.0] * (config.MAX_COINS + 1)
        else:
            initial_action = [0.0] * (len(config.COINS) + 1)
        episode_data["actions"].append(initial_action)
        episode_data["observations"].append(obs.copy())
        
        terminated = truncated = False
        step = 0
        
        while not (terminated or truncated):
            # Get action from the model
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # Apply action to environment
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            
            # Store step data
            episode_data["timestamps"].append(env.current_time)
            episode_data["values"].append(env.current_value)
            episode_data["allocations"].append(env.money_split_ratio.copy())
            episode_data["market_data"].append(env.dfslice.copy())
            episode_data["rewards"].append(reward)
            episode_data["actions"].append(action.copy())
            episode_data["observations"].append(obs.copy())
            
            if render:
                env.render()
                
            # Progress indicator
            if step % 100 == 0:
                print(f"  Step {step}, Portfolio Value: ${env.current_value:.2f}, Reward: {reward:.4f}")
        
        # Consolidate market data into a single DataFrame
        market_data_list = episode_data["market_data"]
        if market_data_list:
            episode_data["market_data"] = pd.concat(market_data_list, ignore_index=True)
        else:
            episode_data["market_data"] = pd.DataFrame()
        
        # Calculate episode statistics
        total_return = episode_data["values"][-1] - episode_data["values"][0]
        percent_return = (total_return / episode_data["values"][0]) * 100
        total_reward = sum(episode_data["rewards"])
        
        print(f"  Episode {episode + 1} completed:")
        print(f"    Steps: {step}")
        print(f"    Total Return: ${total_return:.2f} ({percent_return:.2f}%)")
        print(f"    Total Reward: {total_reward:.4f}")
        print(f"    Final Portfolio Value: ${episode_data['values'][-1]:.2f}")
        
        all_episode_data.append(episode_data)

    return all_episode_data


def extract_meaningful_allocations(episode_data):
    """
    Extract meaningful allocation data from episode data, handling both fixed and variable portfolios.
    
    Args:
        episode_data: Episode data dictionary
        
    Returns:
        tuple: (allocations_df, episode_coins)
    """
    is_variable_portfolio = episode_data.get("is_variable_portfolio", False)
    episode_coins = episode_data.get("episode_coins", config.COINS)
    allocations_raw = episode_data["allocations"]
    timestamps = episode_data["timestamps"]
    
    # Create timezone-naive datetime index
    datetime_index = pd.to_datetime(pd.Index(timestamps)).tz_localize(None)
    
    if is_variable_portfolio:
        # For variable portfolio: extract only meaningful allocations
        meaningful_allocations = []
        
        for allocation_array in allocations_raw:
            # allocation_array has shape (MAX_COINS + 1,) = (6,)
            # [cash, coin1, coin2, coin3, coin4, coin5]
            # But only cash + len(episode_coins) are meaningful
            
            meaningful_allocation = [allocation_array[0]]  # Cash
            
            # Add allocations for active coins
            for i in range(len(episode_coins)):
                meaningful_allocation.append(allocation_array[i + 1])
            
            meaningful_allocations.append(meaningful_allocation)
        
        # Create column names
        column_names = ["Cash"] + episode_coins
        
        # Create DataFrame
        allocations_df = pd.DataFrame(
            meaningful_allocations,
            columns=column_names,
            index=datetime_index
        )
        
        print(f"  📊 Extracted {len(episode_coins)} coin allocations: {episode_coins}")
        
    else:
        # For fixed portfolio: use traditional approach
        allocations_df = pd.DataFrame(
            allocations_raw,
            columns=["Cash"] + config.COINS,
            index=datetime_index
        )
        
        print(f"  📊 Using fixed allocations: {config.COINS}")
    
    return allocations_df, episode_coins


def extract_meaningful_actions(episode_data):
    """
    Extract meaningful action data from episode data, handling both fixed and variable portfolios.
    
    Args:
        episode_data: Episode data dictionary
        
    Returns:
        DataFrame: Actions dataframe with appropriate columns
    """
    is_variable_portfolio = episode_data.get("is_variable_portfolio", False)
    episode_coins = episode_data.get("episode_coins", config.COINS)
    actions_raw = episode_data["actions"]
    timestamps = episode_data["timestamps"]
    
    # Create timezone-naive datetime index
    datetime_index = pd.to_datetime(pd.Index(timestamps)).tz_localize(None)
    
    if is_variable_portfolio:
        # For variable portfolio: extract only meaningful actions
        meaningful_actions = []
        
        for action_array in actions_raw:
            # action_array has shape (MAX_COINS + 1,) = (6,)
            # But only cash + len(episode_coins) are meaningful
            
            meaningful_action = [action_array[0]]  # Cash action
            
            # Add actions for active coins
            for i in range(len(episode_coins)):
                if i + 1 < len(action_array):
                    meaningful_action.append(action_array[i + 1])
                else:
                    meaningful_action.append(0.0)  # Padding if needed
            
            meaningful_actions.append(meaningful_action)
        
        # Create column names
        column_names = ["Cash"] + episode_coins
        
        # Create DataFrame
        actions_df = pd.DataFrame(
            meaningful_actions,
            columns=column_names,
            index=datetime_index
        )
        
    else:
        # For fixed portfolio: use traditional approach
        actions_df = pd.DataFrame(
            actions_raw,
            columns=["Cash"] + config.COINS,
            index=datetime_index
        )
    
    return actions_df


def plot_evaluation_sb3(episode_data, save_plots=True):
    """
    Plot comprehensive evaluation results with SB3 data.
    Compatible with both fixed and variable portfolio environments.
    
    Args:
        episode_data: Episode data dictionary from evaluate_policy_sb3
        save_plots: Whether to save plots to files
    """
    
    timestamps = episode_data["timestamps"]
    portfolio_values = episode_data["values"]
    rewards = episode_data["rewards"]
    
    # Extract meaningful allocation data
    allocations_df, episode_coins = extract_meaningful_allocations(episode_data)
    
    # Create timezone-naive datetime index  
    datetime_index = pd.to_datetime(pd.Index(timestamps)).tz_localize(None)

    portfolio_value_series = pd.Series(portfolio_values, index=datetime_index)
    rewards_series = pd.Series(rewards, index=datetime_index)

    # --- Plot 1: Portfolio Value and Cumulative Rewards ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Portfolio value
    portfolio_value_series.plot(ax=ax1, label='Portfolio Value', color='blue', linewidth=2)
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title('Portfolio Performance Over Episode')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.legend()
    
    # Calculate and display performance metrics
    start_val = portfolio_values[0]
    end_val = portfolio_values[-1]
    percent_return = ((end_val - start_val) / start_val) * 100
    max_val = max(portfolio_values)
    min_val = min(portfolio_values)
    max_drawdown = ((max_val - min_val) / max_val) * 100
    
    performance_text = f'Return: {percent_return:.2f}%\nMax Drawdown: {max_drawdown:.2f}%'
    ax1.text(0.02, 0.95, performance_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7))
    
    # Cumulative rewards
    cumulative_rewards = np.cumsum(rewards)
    ax2.plot(datetime_index, cumulative_rewards, label='Cumulative Reward', color='red', linewidth=2)
    ax2.set_ylabel('Cumulative Reward')
    ax2.set_xlabel('Time')
    ax2.set_title('Cumulative Rewards Over Episode')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.legend()
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_plots:
        output_path = os.path.join(config.LOGDIR, "evaluation_portfolio_performance.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Portfolio performance plot saved to {output_path}")
    plt.show()

    # --- Plot 2: Portfolio Allocation Strategy ---
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    allocations_df.plot.area(ax=ax, stacked=True, linewidth=0, alpha=0.8)
    ax.set_ylabel('Allocation (%)')
    ax.set_xlabel('Time')
    ax.set_title('Portfolio Allocation Strategy Over Time')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(title='Assets', loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.set_ylim(0, 1)
    
    # Add allocation statistics
    avg_allocations = allocations_df.mean()
    allocation_text = "Average Allocations:\n" + "\n".join([f"{asset}: {pct:.1%}" for asset, pct in avg_allocations.items()])
    ax.text(0.02, 0.98, allocation_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_plots:
        output_path = os.path.join(config.LOGDIR, "evaluation_allocation_strategy.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Allocation strategy plot saved to {output_path}")
    plt.show()

    # --- Plot 3: Asset Prices with Allocation Overlay ---
    market_data_df = episode_data["market_data"]
    if not market_data_df.empty:
        market_data_df['date'] = pd.to_datetime(market_data_df['date']).dt.tz_localize(None)

        fig, axes = plt.subplots(len(episode_coins), 1, figsize=(15, 4 * len(episode_coins)), sharex=True)
        if len(episode_coins) == 1:
            axes = [axes]
        
        fig.suptitle('Asset Prices and Allocation Decisions', fontsize=16, y=0.98)

        for ax, coin in zip(axes, episode_coins):
            coin_data = market_data_df[market_data_df['coin'] == coin].set_index('date')
            
            if not coin_data.empty:
                # Plot OHLC prices
                plot_cols = ['open', 'high', 'low', 'close']
                available_cols = [col for col in plot_cols if col in coin_data.columns]
                coin_data[available_cols].plot(ax=ax, linewidth=1.5, alpha=0.8)
                
                ax.set_title(f'{coin} Price Movement')
                ax.set_ylabel('Price ($)')
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
                ax.legend(loc='upper left')

                # Overlay allocation on secondary y-axis
                ax2 = ax.twinx()
                allocations_df[coin].plot(ax=ax2, color='red', linewidth=2, alpha=0.7, label=f'{coin} Allocation')
                ax2.set_ylabel(f'{coin} Allocation', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                ax2.set_ylim(0, 1)
                ax2.legend(loc='upper right')

        axes[-1].set_xlabel('Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_plots:
            output_path = os.path.join(config.LOGDIR, "evaluation_prices_and_allocations.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Asset prices and allocations plot saved to {output_path}")
        plt.show()

    # --- Plot 4: Action Distribution ---
    actions_df = extract_meaningful_actions(episode_data)
    
    plt.figure(figsize=(15, 6))
    ax = plt.gca()
    actions_df.plot(ax=ax, linewidth=1.5, alpha=0.8)
    ax.set_ylabel('Action Values')
    ax.set_xlabel('Time')
    ax.set_title('Raw Action Values Over Time')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(title='Actions', loc='upper right')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_plots:
        output_path = os.path.join(config.LOGDIR, "evaluation_actions.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Actions plot saved to {output_path}")
    plt.show()


def load_sb3_model(model_path, env):
    """
    Load a Stable Baselines 3 model from file.
    
    Args:
        model_path: Path to the saved model
        env: The environment to associate with the model
        
    Returns:
        Loaded SB3 model
    """
    print(f"Loading model from: {model_path}")
    
    # --- FIX: Use custom_objects to specify the custom policy class ---
    # This is crucial for SB3 to know how to reconstruct the model,
    # including its network architecture and optimizers.
    custom_objects = {
        "policy": SimpleMlpTD3Policy
    }
    
    # --- FIX: Make algorithm detection more robust ---
    # Check the parent directory name for the algorithm, and check for TD3 first
    # to avoid the "SAC" in "TRANSACTION_COST" issue.
    run_name_parts = model_path.split(os.sep)
    run_name = ""
    if len(run_name_parts) > 2:
        # e.g., .../models/TD3_tf_agents_style_heavy_.../checkpoints/model.zip -> TD3_TF_AGENTS_STYLE_HEAVY_...
        run_name = run_name_parts[-3].upper()

    model = None
    if "TD3" in run_name:
        print("Detected TD3 model type from path.")
        model = CustomTD3.load(model_path, env=env, custom_objects=custom_objects)
        print("✅ Loaded CustomTD3 model")
    elif "SAC" in run_name:
        print("Detected SAC model type from path.")
        model = SAC.load(model_path, env=env, custom_objects=custom_objects)
        print("✅ Loaded SAC model")
    elif "PPO" in run_name:
        print("Detected PPO model type from path.")
        model = PPO.load(model_path, env=env, custom_objects=custom_objects)
        print("✅ Loaded PPO model")
    else:
        # Fallback if the run name doesn't contain the algorithm
        print("⚠️ Could not detect algorithm from path, trying TD3 as default for this project...")
        try:
            model = CustomTD3.load(model_path, env=env, custom_objects=custom_objects)
            print("✅ Loaded model as CustomTD3 (fallback)")
        except Exception as e:
            print(f"❌ Fallback loading failed: {e}")
            raise

    return model


def print_evaluation_summary(all_episode_data):
    """Print a summary of evaluation results for both fixed and variable portfolios."""
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for i, episode_data in enumerate(all_episode_data):
        start_value = episode_data["values"][0]
        end_value = episode_data["values"][-1]
        total_return = end_value - start_value
        percent_return = (total_return / start_value) * 100
        total_reward = sum(episode_data["rewards"])
        steps = len(episode_data["timestamps"]) - 1
        
        # Get episode-specific information
        episode_coins = episode_data.get("episode_coins", config.COINS)
        is_variable = episode_data.get("is_variable_portfolio", False)
        
        print(f"\nEpisode {i+1} ({'Variable' if is_variable else 'Fixed'} Portfolio):")
        print(f"  Active Assets: {episode_coins}")
        print(f"  Duration: {steps} steps")
        print(f"  Initial Value: ${start_value:.2f}")
        print(f"  Final Value: ${end_value:.2f}")
        print(f"  Total Return: ${total_return:.2f} ({percent_return:.2f}%)")
        print(f"  Total Reward: {total_reward:.4f}")
        print(f"  Avg Reward per Step: {total_reward/steps:.6f}")
        
        # Portfolio allocation summary
        allocations_df, episode_coins = extract_meaningful_allocations(episode_data)
        avg_allocations = allocations_df.mean()
        print(f"  Average Allocations:")
        for asset, allocation in avg_allocations.items():
            print(f"    {asset}: {allocation:.1%}")
    
    # Overall summary
    if len(all_episode_data) > 1:
        all_returns = [((ep["values"][-1] - ep["values"][0]) / ep["values"][0]) * 100 
                      for ep in all_episode_data]
        print(f"\nOverall Performance:")
        print(f"  Mean Return: {np.mean(all_returns):.2f}%")
        print(f"  Std Return: {np.std(all_returns):.2f}%")
        print(f"  Min/Max Return: {np.min(all_returns):.2f}% / {np.max(all_returns):.2f}%")


def test_variable_portfolio_evaluation():
    """Test the evaluation system with variable portfolios."""
    print("🧪 Testing variable portfolio evaluation...")
    
    try:
        # Create test environment
        test_env = PortfolioEnv(use_variable_portfolio=True)
        print(f"✅ Variable portfolio environment created")
        print(f"   Observation space: {test_env.observation_space}")
        print(f"   Action space: {test_env.action_space}")
        
        # Test reset and data collection
        obs, info = test_env.reset()
        episode_coins = info.get('episode_coins', [])
        print(f"✅ Environment reset successful")
        print(f"   Episode coins: {episode_coins}")
        print(f" ?  test_env.money_split_ratio: {test_env.money_split_ratio}")
        print(f"   Money split ratio shape: {test_env.money_split_ratio.shape}")
        
        # Create mock episode data for testing
        mock_episode_data = {
            "timestamps": [test_env.current_time],
            "values": [test_env.current_value],
            "allocations": [test_env.money_split_ratio.copy()],
            "market_data": [test_env.dfslice.copy()],
            "rewards": [0.0],
            "actions": [np.zeros(config.MAX_COINS + 1)],
            "observations": [obs],
            "episode_coins": episode_coins,
            "is_variable_portfolio": True
        }
        
        # Test helper functions
        print("🔧 Testing allocation extraction...")
        allocations_df, extracted_coins = extract_meaningful_allocations(mock_episode_data)
        print(f"✅ Allocation extraction successful")
        print(f"   DataFrame shape: {allocations_df.shape}")
        print(f"   Columns: {list(allocations_df.columns)}")
        print(f"   Extracted coins: {extracted_coins}")
        
        print("🔧 Testing action extraction...")
        actions_df = extract_meaningful_actions(mock_episode_data)
        print(f"✅ Action extraction successful")
        print(f"   DataFrame shape: {actions_df.shape}")
        print(f"   Columns: {list(actions_df.columns)}")
        
        test_env.close()
        print("🎉 Variable portfolio evaluation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Variable portfolio evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # Configuration
    print("🚀 SB3 Portfolio Evaluation Script")
    print("="*50)
    
    # Create log directory if it doesn't exist
    os.makedirs(config.LOGDIR, exist_ok=True)
    
    # Model paths to evaluate
    model_paths = [
        # "models/TD3_tf_agents_style_heavy_20250703_174309/checkpoints/model_92000_steps.zip",
        # "models/TD3_tf_agents_style_heavy_20250703_174309/checkpoints/model_83200_steps.zip",
        # "models/TD3_tf_agents_style_heavy_20250703_174309/final_model.zip",
        
        
        # "models/TD3_tf_agents_style_heavy_20250703_223908/checkpoints/model_100000_steps.zip"
        # "models/TD3_tf_agents_style_heavy_20250703_223908/best_model.zip",
        # "models/TD3_tf_agents_style_light_20250630_201319_final.zip"

        # "models/TD3_tf_agents_style_heavy_TRANSACTION_COST_20250707_133241/checkpoints/model_49200_steps.zip",
        # "models/TD3_tf_agents_style_heavy_TRANSACTION_COST_20250707_133241/checkpoints/model_58400_steps.zip",
        # "models/TD3_tf_agents_style_heavy_TRANSACTION_COST_20250707_133241/final_model.zip"

        # "models/TD3_tf_agents_style_heavy_TRANSACTION_COST_20250711_130642/final_model.zip"

        # "models/TD3_tf_agents_style_heavy_STRUCTURED_CREDIT_20250714_190911/checkpoints/model_70400_steps.zip"
        # "models/TD3_tf_agents_style_heavy_STRUCTURED_CREDIT_20250714_194641/checkpoints/model_57200_steps.zip"
        # "models/TD3_tf_agents_style_heavy_STRUCTURED_CREDIT_20250714_153933/checkpoints/model_62800_steps.zip"
        # "models/TD3_tf_agents_style_heavy_SHAPLEY_20250719_141411/checkpoints/model_40000_steps.zip"
    

        "models/TD3_tf_agents_style_heavy_POMDP_20250720_022055/checkpoints/model_62400_steps.zip"

    ]
    
    # Check which models exist
    available_models = []
    for model_path in model_paths:
        if os.path.exists(model_path):
            available_models.append(model_path)
            print(f"✅ Found model: {model_path}")
        else:
            print(f"❌ Model not found: {model_path}")
    
    if not available_models:
        print("❌ No models found! Please train a model first.")
        exit(1)
    
    # Use the first available model
    model_path = available_models[0]
    print(f"\n📊 Evaluating model: {model_path}")
    
    # --- FIX: Create environment BEFORE loading the model ---
    print("🏗️  Creating evaluation environment...")
    # Infer reward type from model path to ensure consistency
    reward_type = "TRANSACTION_COST" if "TRANSACTION_COST" in model_path.upper() else "simple"
    eval_env = PortfolioEnv(reward_type=reward_type)
    print(f"✅ Environment created with reward_type='{reward_type}'")
    print(f"   Observation space: {eval_env.observation_space}")
    print(f"   Action space: {eval_env.action_space}")

    # --- FIX: Load the model AFTER creating the environment ---
    model = None  # Initialize model to None
    try:
        model = load_sb3_model(model_path, env=eval_env)
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Run evaluation
    print(f"\n🧪 Running evaluation...")
    num_episodes = 1  # Change this to run multiple episodes if desired
    evaluation_data = evaluate_policy_sb3(
        model, eval_env, 
        num_episodes=num_episodes, 
        deterministic=True,
        render=False
    )
    
    # Plot results
    print(f"\n📈 Generating plots...")
    plot_evaluation_sb3(evaluation_data[0], save_plots=True)
    
    # Print summary
    print_evaluation_summary(evaluation_data)
    
    print(f"\n🎉 Evaluation completed!")
    print(f"📁 Plots saved to: {config.LOGDIR}")
    
    # Close environment
    eval_env.close()

    # Test variable portfolio evaluation
    test_variable_portfolio_evaluation()

# %%

