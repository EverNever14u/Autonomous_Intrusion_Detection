from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import shutil
import time
import os


# ANSI color codes
class Color:
    GREEN = '\033[92m'
    RESET = '\033[0m'

# Function to format time in green
def get_duration(start):
    duration_s = (time.time() - start)
    return f"{Color.GREEN}({duration_s:.2f}s){Color.RESET}"

# Data import
start_time = time.time()
PATH = "data/CICIDS2017_improved/friday.csv" 
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
    test_size=0.2, 
    random_state=42, 
    stratify=y
)
print(f"Data division done {get_duration(start_time)}")

print("-" * 30)
print(f"Data processing complete")
print(f"Training rows: {X_train.shape[0]}")
print(f"Test rows: {X_test.shape[0]}")



# CLEANED_DIR = "data/cleaned"

 
# start_time = time.time()
# if os.path.exists(CLEANED_DIR):
#     shutil.rmtree(CLEANED_DIR)  
# os.makedirs(CLEANED_DIR)        
# print(f"Folder {CLEANED_DIR} cleared and recreated {get_duration(start_time)}")


# X_train_df = pd.DataFrame(X_train, columns=X.columns)
# X_test_df = pd.DataFrame(X_test, columns=X.columns)


# start_time = time.time()
# X_train_df.to_csv(f"{CLEANED_DIR}/X_train.csv", index=False)
# X_test_df.to_csv(f"{CLEANED_DIR}/X_test.csv", index=False)
# y_train.to_csv(f"{CLEANED_DIR}/y_train.csv", index=False)
# y_test.to_csv(f"{CLEANED_DIR}/y_test.csv", index=False)

# print(f"All files saved{CLEANED_DIR} {get_duration(start_time)}")
# print("-" * 30)

print(f"{Color.GREEN}FIN{Color.RESET}")



#plots


plt.figure(figsize=(10, 5)) 
label_names = le.inverse_transform(y)
sns.countplot(
    y=label_names, 
    order=pd.Series(label_names).value_counts().index, 
    palette='viridis'
)
plt.title('Attacks vs Normal Traffic')
plt.xlabel('Number of samples')
plt.ylabel('Traffic type')
plt.tight_layout()



feature_idx = 1 
feature_name = X.columns[feature_idx]

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(X.iloc[:, feature_idx], bins=50, ax=axes2[0], color='orange', kde=True)
axes2[0].set_title(f'Before scaling: {feature_name}')
axes2[0].set_ylabel('Count')

sns.histplot(X_scaled[:, feature_idx], bins=50, ax=axes2[1], color='blue', kde=True)
axes2[1].set_title(f'After scaling: {feature_name}')
axes2[1].set_ylabel('Count')

plt.tight_layout()
plt.show()


