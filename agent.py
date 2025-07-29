import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import math


class SoccerBettingEnv(gym.Env):
    def __init__(self, data):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.current_step = 0

        # Observation space: HomeELO, AwayELO, PSH, PSD, PSA, MaxH, MaxD, MaxA, P>2.5, P<2.5, Max>2.5, Max<2.5, AHh, PAHH, PAHA, MaxAHH, MaxAHA
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        # Action: [home_bet, draw_bet, away_bet, over_bet, under_bet, home_spread_bet, away_spread_bet] ∈ [0, 1], capped to sum ≤ 1
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.data = self.data.sample(frac=1.0).reset_index(drop=True)
        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.data.iloc[self.current_step]
        return np.array([
            row["HomeELO"],
            row["AwayELO"],
            row["PSH"],
            row["PSD"],
            row["PSA"],
            row["P>2.5"],
            row["P<2.5"],
            row["AHh"],
            row["PAHH"],
            row["PAHA"],
        ], dtype=np.float32)

    def step(self, action):
        row = self.data.iloc[self.current_step]

        # Normalize action to ensure total bet ≤ 1 unit
        action = np.clip(action, 0.0, 1.0)
        total_bet = np.sum(action)
        if total_bet > 1.0:
            action = action / total_bet
            
        home_bet, draw_bet, away_bet, over_bet, under_bet, home_spread_bet, away_spread_bet = action
        reward = 0

        if row["FTHG"] > row["FTAG"]:
            reward += home_bet * (row["PSH"] - 1)
            reward -= draw_bet
            reward -= away_bet
        elif row["FTHG"] == row["FTAG"]:
            reward += draw_bet * (row["PSD"] - 1)
            reward -= home_bet
            reward -= away_bet
        elif row["FTHG"] < row["FTAG"]:
            reward += away_bet * (row["PSA"]- 1)
            reward -= home_bet
            reward -= draw_bet

        if row["FTHG"] + row["FTAG"] > 2.5:
            reward += over_bet * (row["P>2.5"]-1)
            reward -= under_bet
        else:
            reward += under_bet * (row["P<2.5"]-1)
            reward -= over_bet

        _, decimal_part = math.modf(row["AHh"])
        if decimal_part in (0, 0.5):
            if row["FTHG"] + row["AHh"] > row["FTAG"]:
                reward += home_spread_bet * (row["PAHH"] - 1)
                reward -= away_spread_bet
            elif row["FTHG"] + row["AHh"] < row["FTAG"]:
                reward += away_spread_bet * (row["PAHA"] - 1)
                reward -= home_spread_bet
        else:
            if row["FTHG"] + row["AHh"] + 0.25 > row["FTAG"]:
                reward += home_spread_bet/2 * (row["PAHH"] - 1)
                reward -= away_spread_bet/2
            elif row["FTHG"] + row["AHh"] + 0.25 < row["FTAG"]:
                reward += away_spread_bet/2 * (row["PAHA"] - 1)
                reward -= home_spread_bet/2

            if row["FTHG"] + row["AHh"] - 0.25 > row["FTAG"]:
                reward += home_spread_bet/2 * (row["PAHH"] - 1)
                reward -= away_spread_bet/2
            elif row["FTHG"] + row["AHh"] - 0.25 < row["FTAG"]:
                reward += away_spread_bet/2 * (row["PAHA"] - 1)
                reward -= home_spread_bet/2


        self.current_step += 1
        done = self.current_step >= len(self.data)
        observation = self._get_obs() if not done else np.zeros(5, dtype=np.float32)
        info = {}

        return observation, reward, done, False, info

    def render(self):
        pass  # Optional visualization

# Load dataset
df = pd.read_csv("master_data.csv")  # Needs proper formatting and headers
print("Dataset loaded. Initializing environment")

# Create and validate the environment
env = SoccerBettingEnv(df)
check_env(env, warn=True)  # Optional sanity check

# Train the agent
model = PPO("MlpPolicy", env, verbose=1)

increment = 10_000
best_profit = -10000

while True:
    env = SoccerBettingEnv(df)
    model.set_env(env)
    model.learn(total_timesteps=increment, reset_num_timesteps=False)

    obs, _ = env.reset()
    total_reward = 0
    total_bets = {"Home": 0.0, "Draw": 0.0, "Away": 0.0, "Over": 0.0, "Under": 0.0, "Home Spread": 0.0, "Away Spread": 0.0, "No Bet": 0}

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward

        action = np.clip(action, 0.0, 1.0)
        total = np.sum(action)

        if total < 1e-3:
            total_bets["No Bet"] += 1
        else:
            total_bets["Home"] += float(action[0])
            total_bets["Draw"] += float(action[1])
            total_bets["Away"] += float(action[2])
            total_bets["Over"] += float(action[3])
            total_bets["Under"] += float(action[4])
            total_bets["Home Spread"] += float(action[5])
            total_bets["Away Spread"] += float(action[6])            
        
        if done or truncated:
            break
        
    print(f"Best Profit: {best_profit:.2f}")
    print(f"Total Profit: {total_reward:.2f}")
    print("Total Bet Amounts:")
    for k, v in total_bets.items():
        print(f"  {k}: {v:.2f}")

    if total_reward > best_profit:
        best_profit = total_reward
        print("Saving best run so far!")
        model.save("soccer_betting_agent")

        
