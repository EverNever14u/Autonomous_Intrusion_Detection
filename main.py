from tools import *
from plot_generator import generate_plots
from rl_environment import IDS_Environment
from data_saver import save_processed_data

from stable_baselines3.common.evaluation import evaluate_policy
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from stable_baselines3 import PPO
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import shutil
import time
import os





# Data import
start_time = time.time()

load_dotenv(override=True)
PATH = os.getenv("DATA_PATH", "data/CICIDS2017_improved/monday.csv")
CLEANED_DIR = os.getenv("CLEANED_DIR", "data/cleaned")
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))

SHOULD_SAVE = str_to_bool(os.getenv("SAVE_DATA", "True"))
SHOULD_PLOT = str_to_bool(os.getenv("SHOW_PLOTS", "True"))

try:
    data = pd.read_csv(PATH)
    print(f"Loaded {data.shape[0]} rows {get_duration(start_time)}.")
except FileNotFoundError:
    print("File not found. Please check your PATH.")
    exit()

# Delete meaningless columns
start_time = time.time()
to_drop = ['id', 'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp', 'Attempted Category']
data = data.drop(columns=[column for column in to_drop if column in data.columns])
print(f"Column deletion done {get_duration(start_time)}")

# Label encoding
start_time = time.time()
le = LabelEncoder()
data['Label'] = le.fit_transform(data['Label'])
print(f"Classes encoded {get_duration(start_time)}")

# Feature/Target separation
start_time = time.time()
X = data.drop('Label', axis=1)
y = data['Label']
print(f"Dropping done {get_duration(start_time)}")

# Scaling
start_time = time.time()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Scaling done {get_duration(start_time)}")

# Data division
start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE, 
    stratify=y
)
print(f"Data division done {get_duration(start_time)}")


print(f"Training rows: {Color.BLUE}{X_train.shape[0]}{Color.RESET}")
print(f"Test rows: {Color.BLUE}{X_test.shape[0]}{Color.RESET}")

# saving clean data
if SHOULD_SAVE: save_processed_data(X_train, X_test, y_train, y_test, X.columns, CLEANED_DIR)
else: print(f"Skiped saving {Color.BLUE}(due to config){Color.RESET}")
#plots
if SHOULD_PLOT: generate_plots(X, X_scaled, y, le)
else: print(f"Skiped generating plot {Color.BLUE}(due to config){Color.RESET}")


print(f"{Color.GREEN}\nData procesing done{Color.RESET}")
print(f"{Color.GREEN}-{Color.RESET}" * 30 + "\n")


#rl_environment
start_time = time.time()
env = IDS_Environment(X_train, y_train)
print(f"init RL env done {get_duration(start_time)}")

start_time = time.time()
model = PPO("MlpPolicy", env, verbose=1)
print(f"init model done {get_duration(start_time)}")

start_time = time.time()
model.learn(total_timesteps=10000) 
print(f"Trening done {get_duration(start_time)}")

test_env = IDS_Environment(X_test, y_test)

obs, info = test_env.reset()
total_reward = 0
correct_predictions = 0
STEPS_TO_TEST = 1000


for i in range(STEPS_TO_TEST):
   
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    total_reward += reward
    
    true_label = y_test.iloc[i] if isinstance(y_test, pd.Series) else y_test[i]
    

    if action == true_label:
        correct_predictions += 1
        
    if i < 10:
        result_color = Color.GREEN if action == true_label else Color.RED
        print(f"Test step {i+1}: AI action={action} Truth={true_label} Reward={result_color}{reward}{Color.RESET}")
    
    if terminated:
        obs, info = test_env.reset()


accuracy = (correct_predictions / STEPS_TO_TEST) * 100
print(f"\n{Color.BLUE}--- Test results ---{Color.RESET}")
print(f"packages tested: {Color.BLUE}{STEPS_TO_TEST}{Color.RESET}")
print(f"accuracy: {Color.BLUE}{accuracy:.2f}%{Color.RESET}")