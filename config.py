# Configuration file for portfolio allocation RL environment
# Compatible with Stable Baselines 3 and Gymnasium

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Data file path
FILE = "cleaned_preprocessed.csv"

# Original cryptocurrency coins (for backward compatibility)
COINS = ["DASH", "LTC", "STR"]

# =============================================================================
# VARIABLE PORTFOLIO CONFIGURATION
# =============================================================================

# Portfolio allocation mode (moved here to be used early)
USE_VARIABLE_PORTFOLIO = False  # Set to False for backward compatibility

# Maximum number of coins supported in a single episode
MAX_COINS = 5  # Reduced from 10 for realistic testing

# Extended coin pool to choose from (only coins available in dataset)
# Note: For demonstration of variable portfolio, we'll use subsets of available coins
COIN_POOL = ["DASH", "LTC", "STR"]

# Range of coins to select per episode (min, max)
COINS_PER_EPISODE_RANGE = (3, 3) if not USE_VARIABLE_PORTFOLIO else (1, 3)  # Fixed 3 coins for stability

# Raw columns in the dataset
COLS = ['high', 'low', 'open', 'close', 'volume', 'quoteVolume', 'weightedAverage']

# ✅ CONVERGENCE FIX: Use ONLY 30-day rolling averages like successful implementation
# This reduces complexity from 84 to 42 features, matching the working portfolio-optimization-main
SCOLS_SIMPLIFIED = ['vh', 'vl', 'vc', 'open_s', 'volume_s', 'quoteVolume_s', 'weightedAverage_s', 
                    'vh_roll_30', 'vl_roll_30', 'vc_roll_30', 'open_s_roll_30', 'volume_s_roll_30', 
                    'quoteVolume_s_roll_30', 'weightedAverage_s_roll_30']

# ✅ RESTORED: Full feature set with 7, 14, and 30-day rolling averages (28 features per coin)
# As requested, this restores the original, more complex feature set.
BASE_SCOLS = ['vh', 'vl', 'vc', 'open_s', 'volume_s', 'quoteVolume_s', 'weightedAverage_s']
ROLLING_WINDOWS = ['7', '14', '30']
SCOLS = list(BASE_SCOLS)
for col in BASE_SCOLS:
    for window in ROLLING_WINDOWS:
        SCOLS.append(f'{col}_roll_{window}')

# Features per coin (calculated after SCOLS is defined)
FEATURES_PER_COIN = len(SCOLS)  # Now 28 features per coin (was 14)

# =============================================================================
# TEMPORAL SMOOTHNESS CONFIGURATION (for POMDP reward-based smoothing)
# =============================================================================

# Temporal smoothness penalty weight (applied in reward function, not action space)
TEMPORAL_SMOOTHNESS_WEIGHT = 1.0  # Weight for L2 penalty on allocation changes

# Smoothing window length 
SMOOTHING_WINDOW = 30  # Number of previous steps to consider for smoothness

# # Volatility-adjusted smoothing
# VOLATILITY_SCALING = False  # Enable volatility-adjusted smoothness penalty
# VOLATILITY_THRESHOLD = 0.02  # 2% volatility threshold for scaling

# # Smoothness penalty scaling factors
# SMOOTHNESS_MIN_SCALE = 0.5  # Minimum scaling during high volatility
# SMOOTHNESS_MAX_SCALE = 2.0  # Maximum scaling during low volatility

# Original observation columns (for backward compatibility)
OBS_COLS = ['DASH_vh', 'LTC_vh', 'STR_vh', 'DASH_vl', 'LTC_vl', 'STR_vl', 'DASH_vc', 'LTC_vc', 'STR_vc',
            'DASH_open_s', 'LTC_open_s', 'STR_open_s', 'DASH_volume_s', 'LTC_volume_s', 'STR_volume_s', 
            'DASH_quoteVolume_s', 'LTC_quoteVolume_s', 'STR_quoteVolume_s', 'DASH_weightedAverage_s', 
            'LTC_weightedAverage_s', 'STR_weightedAverage_s', 'DASH_vh_roll_30', 'LTC_vh_roll_30', 'STR_vh_roll_30',
            'DASH_vl_roll_30', 'LTC_vl_roll_30', 'STR_vl_roll_30', 'DASH_vc_roll_30', 'LTC_vc_roll_30', 
            'STR_vc_roll_30', 'DASH_open_s_roll_30', 'LTC_open_s_roll_30', 'STR_open_s_roll_30', 
            'DASH_volume_s_roll_30', 'LTC_volume_s_roll_30', 'STR_volume_s_roll_30', 'DASH_quoteVolume_s_roll_30',
            'LTC_quoteVolume_s_roll_30', 'STR_quoteVolume_s_roll_30', 'DASH_weightedAverage_s_roll_30', 
            'LTC_weightedAverage_s_roll_30', 'STR_weightedAverage_s_roll_30']

# Variable portfolio observation space configuration
MAX_OBSERVATION_DIM = MAX_COINS * FEATURES_PER_COIN  # Maximum possible observation dimension

# ⚠️ PLACEHOLDER: Min/max values for the restored 28-feature set
# The original min/max values for this expanded feature set are not available.
# These are placeholders and MUST be replaced with the correct, calculated values.
SINGLE_COIN_FEATURE_MIN = [-100.0] * FEATURES_PER_COIN
SINGLE_COIN_FEATURE_MAX = [100.0] * FEATURES_PER_COIN

# Ensure we have the right number of min/max values (should be 28 now)
assert len(SINGLE_COIN_FEATURE_MIN) == FEATURES_PER_COIN, f"Min features length mismatch: {len(SINGLE_COIN_FEATURE_MIN)} != {FEATURES_PER_COIN}"
assert len(SINGLE_COIN_FEATURE_MAX) == FEATURES_PER_COIN, f"Max features length mismatch: {len(SINGLE_COIN_FEATURE_MAX)} != {FEATURES_PER_COIN}"

print(f"✅ RESTORED: Increased features per coin from 14 to {FEATURES_PER_COIN}")
print(f"✅ Total observation dimension: {MAX_OBSERVATION_DIM} (was {MAX_COINS * 14})")
print(f"⚠️ WARNING: Using placeholder min/max values for observation space. These must be updated.")

# Generate min/max arrays for maximum possible coins
OBS_COLS_MIN_VARIABLE = SINGLE_COIN_FEATURE_MIN * MAX_COINS
OBS_COLS_MAX_VARIABLE = SINGLE_COIN_FEATURE_MAX * MAX_COINS

# Legacy min/max values for backward compatibility
# NOTE: These are for the 42-feature set (14 features x 3 coins) and are no longer used
# by the variable portfolio environment with the restored feature set.
OBS_COLS_MIN = [-0.39,-0.39,-0.39,-52.92,-144.06,-159.81,-31.78,-19.57,-61.76,-87.7,-65.71,-130.35,-51.36,-51.09,-77.06,-0.03,-0.16,-74.55,-73.04,-73.72,-145.94,-0.77,-0.77,-0.77,-12.17,-14.24,-25.47,-15.64,-12.37,-12.73,-44.75,-25.01,-32.13,-39.2,-83.44,-74.04,-0.03,-0.2,-75.79,-44,-25,-32.25]
OBS_COLS_MAX = [53.16,69.71,90.97,0.4,0.4,0.4,39.31,24.4,85.49,114.25,66.32,130.35,55.72,60.25,66.09,0.04,0.18,96.32,67.32,73.63,145.94,9.96,17.79,53.89,0.81,0.81,0.81,7.58,9.9,14.93,41.44,16.35,32.11,34.64,82.48,59.62,0.03,0.2,78.16,38.94,16.29,32.24]

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

# Episode length (number of time steps per episode)
EPISODE_LENGTH = 1500

# Logging and model save directories
LOGDIR = "logs"
MODEL_SAVE = "models"

# =============================================================================
# STABLE BASELINES 3 HYPERPARAMETERS
# =============================================================================

# Training configuration
NUM_ITERATIONS = 1000
TOTAL_TIMESTEPS = 100000
EVAL_FREQ = 4  # More frequent evaluation like the successful implementation
LOG_INTERVAL = 10
MODEL_SAVE_FREQ = 12

# Evaluation settings
NUM_EVAL_EPISODES = 4  # Reduced like the successful implementation

# =============================================================================
# PPO HYPERPARAMETERS
# =============================================================================
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 1024,        # Reduced for more frequent updates
    "batch_size": 100,      # Aligned with successful implementation
    "n_epochs": 10,
    "gamma": 0.05,         # Lower gamma like successful implementation
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}

# =============================================================================
# SAC HYPERPARAMETERS  
# =============================================================================
SAC_CONFIG = {
    "learning_rate": 3e-4,
    "buffer_size": 10000,   # Smaller buffer like successful implementation
    "learning_starts": 1000,
    "batch_size": 100,      # Aligned with successful implementation
    "tau": 0.05,           # From successful implementation
    "gamma": 0.05,         # Lower gamma like successful implementation
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",
}

# =============================================================================
# TD3 HYPERPARAMETERS
# =============================================================================
TD3_CONFIG = {
    "learning_rate": 3e-4,
    "buffer_size": 10000,   # Smaller buffer like successful implementation
    "learning_starts": 1000,
    "batch_size": 100,      # Aligned with successful implementation
    "tau": 0.05,           # From successful implementation
    "gamma": 0.05,         # Lower gamma like successful implementation
    "train_freq": 1,
    "gradient_steps": 1,
    "policy_delay": 2,
    "target_policy_noise": 0.2,
    "target_noise_clip": 0.5,
}

# =============================================================================
# DDPG HYPERPARAMETERS
# =============================================================================
DDPG_CONFIG = {
    "learning_rate": 1e-4,  # Actor learning rate from successful implementation
    "buffer_size": 10000,   # Smaller buffer like successful implementation
    "learning_starts": 1000,
    "batch_size": 100,      # Batch size from successful implementation
    "tau": 0.05,           # Target update tau from successful implementation
    "gamma": 0.05,         # Lower gamma like successful implementation
    "train_freq": 1,
    "gradient_steps": 1,
    "action_noise": None,
}

# =============================================================================
# ATTENTION MECHANISM CONFIGURATION
# =============================================================================

# Number of coins (for CoinAttentionExtractor) - now variable
N_COINS = len(COINS)  # Legacy compatibility
N_COINS_VARIABLE = MAX_COINS  # For variable portfolio

# Features per coin (for reshaping observations) - now standardized
FEATURES_PER_COIN_LEGACY = len(OBS_COLS) // N_COINS

# Attention configurations - updated for variable portfolio
ATTENTION_CONFIGS = {
    "light": {
        "features_dim": 128,
        "n_heads": 4,
        "n_layers": 1,
        "dropout": 0.1,
        "max_coins": MAX_COINS,
        "features_per_coin": FEATURES_PER_COIN  # Now correctly uses 14 features
    },
    "medium": {
        "features_dim": 256,
        "n_heads": 8,
        "n_layers": 2,
        "dropout": 0.1,
        "max_coins": MAX_COINS,
        "features_per_coin": FEATURES_PER_COIN  # Now correctly uses 14 features
    },
    "heavy": {
        "features_dim": 512,
        "n_heads": 16,  # Changed from 12 to 16 (512 ÷ 16 = 32)
        "n_layers": 3,
        "dropout": 0.2,
        "max_coins": MAX_COINS,
        "features_per_coin": FEATURES_PER_COIN  # Now correctly uses 14 features
    }
}

# =============================================================================
# PORTFOLIO ALLOCATION SPECIFIC SETTINGS
# =============================================================================

# Initial portfolio value
INITIAL_CASH = 1000

# Time delta between steps (5 minutes)
TIME_DELTA_MINUTES = 5

# Reward scaling factor
REWARD_SCALE_FACTOR = 1.0

# Transaction cost (as percentage)
TRANSACTION_COST = 0.02

# Volatility penalty weight
VOLATILITY_PENALTY_WEIGHT = 1.0  # Add this: Controls the penalty for large allocation changes.
# A higher value encourages smoother, more stable allocation strategies.

# Long-term performance bonus parameters
LONG_TERM_BONUS_ENABLED = True  # Enable/disable long-term performance bonus
LONG_TERM_LAMBDA = 2.0  # λ_long: Weight for long-term performance bonus (increased from 1.5)
LONG_TERM_LOOKBACK = 30  # N: Number of steps to look back for long-term comparison
# Formula: r_t' = r_t + λ_long * (1/N) * (V_t - V_{t-N}) / V_{t-N}

# --- NEW: Shapley Value Parameters ---
SHAPLEY_SAMPLES = 64  # Number of Monte Carlo samples for Shapley value calculation

# --- NEW: Structured Reward Parameters ---
STRUCTURED_REWARD_SCALING_FACTOR = 5.0 # Beta for tanh squashing of returns

# --- NEW: Total Variation Penalty for Thrashing ---
TV_WINDOW = 15                  # Number of past steps to consider for thrashing
TV_WEIGHT = 1.5                 # Reduced weight to balance with stronger performance signals
TV_BASELINE_VARIATION = 0.02    # Increased baseline: Allow more normal market-responsive changes

# --- NEW: Allocation Volatility Penalty ---
ALLOCATION_VOLATILITY_WEIGHT = 1.0 # Weight for penalty on the std dev of allocation changes

# --- REVISED: Symmetric Drawdown Reward/Penalty (Tuned for Intraday Trading) ---
# This provides a continuous reward for staying near the peak portfolio value
# and a penalty for falling away from it.
DRAWDOWN_WINDOW = 30            # Lookback window to calculate the peak portfolio value
DRAWDOWN_REWARD_WEIGHT = 2.0    # Increased weight to make this signal stronger
DRAWDOWN_CROSSOVER = 0.015      # Crossover at 1.5% drawdown. This is now highly sensitive.
DRAWDOWN_EXPONENT = 1.5         # Softened exponent slightly to create a more linear response

# Type of reward function to use: "simple", "TRANSACTION_COST", "STRUCTURED_CREDIT", "POMDP"
REWARD_TYPE = "POMDP"

# --- NEW: Per-Asset Credit Assignment (for POMDP) ---
# This provides a reward gradient to incentivize allocating to high-performing but ignored assets.
CREDIT_ASSIGNMENT_WEIGHT = 0.02 # Weight for the correlation-based opportunity cost reward

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_algorithm_config(algorithm_name):
    """Get hyperparameter configuration for specified algorithm."""
    configs = {
        "PPO": PPO_CONFIG,
        "SAC": SAC_CONFIG,
        "TD3": TD3_CONFIG,
        "DDPG": DDPG_CONFIG
    }
    return configs.get(algorithm_name.upper(), PPO_CONFIG)

def get_attention_config(config_name="medium"):
    """Get attention configuration by name."""
    return ATTENTION_CONFIGS.get(config_name, ATTENTION_CONFIGS["medium"])

def get_random_coin_subset():
    """Get a random subset of coins for variable portfolio training."""
    import random
    min_coins, max_coins = COINS_PER_EPISODE_RANGE
    n_coins = random.randint(min_coins, min(max_coins, len(COIN_POOL)))
    return random.sample(COIN_POOL, n_coins)

def create_variable_obs_columns(selected_coins):
    """Create observation column names for selected coins."""
    obs_cols = []
    for coin in selected_coins:
        for scol in SCOLS:
            obs_cols.append(f"{coin}_{scol}")
    return obs_cols

def validate_config():
    """Validate that configuration is consistent."""
    if USE_VARIABLE_PORTFOLIO:
        assert MAX_COINS >= max(COINS_PER_EPISODE_RANGE), \
            "MAX_COINS must be >= maximum coins per episode"
        
        assert all(coin in COIN_POOL for coin in COINS), \
            "All legacy COINS must be in COIN_POOL"
        
        assert len(SINGLE_COIN_FEATURE_MIN) == FEATURES_PER_COIN, \
            "Single coin feature bounds must match FEATURES_PER_COIN"
        
        print(f"✅ Variable portfolio configuration validated")
        print(f"📊 Max coins: {MAX_COINS}, Features per coin: {FEATURES_PER_COIN}")
        print(f"🎲 Coins per episode: {COINS_PER_EPISODE_RANGE[0]}-{COINS_PER_EPISODE_RANGE[1]}")
        print(f"🪙 Coin pool: {COIN_POOL}")
    else:
        # Legacy validation
        assert len(OBS_COLS) == len(OBS_COLS_MIN) == len(OBS_COLS_MAX), \
            "Observation columns and bounds must have same length for legacy mode"
        
        assert len(OBS_COLS) % N_COINS == 0, \
            "Number of observation features must be divisible by number of coins"
        
        print(f"✅ Legacy configuration validated")
        print(f"📊 Data: {len(COINS)} coins, {len(OBS_COLS)} features ({FEATURES_PER_COIN_LEGACY} per coin)")
    
    print(f"🏋️  Episode length: {EPISODE_LENGTH} steps")
    print(f"💰 Initial cash: ${INITIAL_CASH}")

if __name__ == "__main__":
    validate_config()