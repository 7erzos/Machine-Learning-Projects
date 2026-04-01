import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

path = '/content/drive/MyDrive/lanl-auth-dataset-1-00.csv'

df = pd.read_csv(
                   path,
                   header=None, # Prevent the first data row from being column names
                   names=["time","user","host"], # Assigning Headers
                   usecols=[0, 1, 2], # In Case if any lines have any extra commas
                   sep=",",
                   engine="c",
                   on_bad_lines="skip" # Skipping lines that create NaN
                  ).dropna() # Enforce that every event is a complete tripple


# Note that we will have to handle missing Values
df = df.dropna(subset=["time","user","host"]).copy()

# Enforcing Types to the Data
df["time"] = df["time"].astype(np.int64)
df["user"] = df["user"].astype("string")
df["host"] = df["host"].astype("string")

df.head(), df.tail(), df.shape

print(df.isna().sum())
print(df["user"].head(3).tolist(), df["host"].head(3).tolist())
print("unique users:", df["user"].nunique(), "unique hosts:", df["host"].nunique())
print("min time:", df["time"].min(), "max time:", df["time"].max())

# Using Groupby instead of a loop
# - groupby().size() is vectorized C-Speed
# - A python loop over 3M rows is slow and prone to errors

counts = df.groupby(["user","host"]).size().reset_index(name="cnt") # Since the model does not directly use individual events. It uses the sufficient statistic for association strength: How often u hits h

counts.head(), counts.shape


# Subsetting (Colab Constraints + Modeling Reason)

# 1. Memory: Full Matrix would be too dense which is large
# 2. Signal: Users/Hosts with tiny activity are mostly noise for low-rank learning

top_users = 3000
top_hosts = 4000
min_cnt = 3


counts = counts[counts["cnt"] >= min_cnt]

u_freq = counts.groupby("user")["cnt"].sum().sort_values(ascending=False) # User Activity Mass
h_freq = counts.groupby("host")["cnt"].sum().sort_values(ascending=False) # Host Popularity Mass

# Use Set to do membership checks:
# - x in set is ~O(1) Average
# ~ x in list is O(n)
keep_u = set(u_freq.head(top_users).index)
keep_h = set(h_freq.head(top_hosts).index)

sub = counts[counts["user"].isin(keep_u) & counts["host"].isin(keep_h)].copy()

sub.shape

# Building my X Matrix: Edge List --> Dense Array

# Index Sets
users = sub["user"].unique()
hosts = sub["host"].unique()

# Dictionaries for My Matrix
u2i = {u:i for i,u in enumerate(users)}
h2j = {h:j for j,h in enumerate(hosts)}

# Allocating the Numeric Object which the Model Expects
m, n = len(users), len(hosts)

# Setting the X Matrix
X = np.zeros((m, n), dtype=np.float32)

# Creating Bijections and Mapping
ui = sub["user"].map(u2i).to_numpy()
hj = sub["host"].map(h2j).to_numpy()
cnt = sub["cnt"].to_numpy(dtype=np.float32)

# Using Log1p Formula due to counts having a long tail. Defines the Interaction Strength
X[ui, hj] = np.log1p(cnt)

# 1. Get the coordinates of non-zero entries (interactions)
rows, cols = X.nonzero()
vals = X[rows, cols]

# 2. Split the interactions into training and testing sets
# We split the indices of the non-zero entries
idx_train, idx_test = train_test_split(np.arange(len(rows)), test_size=0.2, random_state=42)

# 3. Create a training matrix
X_train = np.zeros_like(X)
X_train[rows[idx_train], cols[idx_train]] = vals[idx_train]

# 4. Store the test pairs for evaluation
test_rows = rows[idx_test]
test_cols = cols[idx_test]
test_vals = vals[idx_test]

print(f"Training interactions: {len(idx_train)}")
print(f"Testing interactions: {len(idx_test)}")

k = 50
n_iter = 10
print(f'Hyperparameters defined: k={k}, n_iter={n_iter}')

svd = TruncatedSVD(n_components=k, n_iter=n_iter, random_state=0)
U = svd.fit_transform(X_train)
V = svd.components_
scores = U @ V

print(f'U shape: {U.shape}')
print(f'V shape: {V.shape}')
print(f'Scores matrix shape: {scores.shape}')

from sklearn.metrics import mean_squared_error

# Extract predicted scores for the test pairs
predicted_test_vals = scores[test_rows, test_cols]

# Calculate MSE
mse = mean_squared_error(test_vals, predicted_test_vals)
rmse = np.sqrt(mse)

print(f"Test MSE: {mse:.4f}")
print(f"Test RMSE: {rmse:.4f}")

# Quick check on a few predictions
for i in range(5):
    print(f"Actual: {test_vals[i]:.2f}, Predicted: {predicted_test_vals[i]:.2f}")

# 1. Create a mapping of users to the hosts they actually interacted with (Ground Truth)
# We'll use the original 'sub' dataframe for this
by_user = sub.groupby('user')['host'].apply(set).to_dict()

# 2. Pick a sample user to evaluate (e.g., the first user in our index)
target_user_name = users[0]
target_user_idx = u2i[target_user_name]
true_hosts = by_user[target_user_name]

# 3. Get model recommendations for this user
# We get the scores for all hosts for this user
user_scores = scores[target_user_idx]

# 4. Find Top-K recommendations (excluding hosts they already visited in training)
K = 10
top_indices = np.argsort(user_scores)[::-1]
rec_hosts = [hosts[idx] for idx in top_indices[:K]]

# 5. Calculate Recall@K
hits = len(set(rec_hosts) & true_hosts)
recall = hits / len(true_hosts)

print(f"Evaluation for user: {target_user_name}")
print(f"Actual hosts visited: {len(true_hosts)}")
print(f"Top-{K} Recommendations: {rec_hosts}")
print(f"Hits: {hits}")
print(f"Recall@{K}: {recall:.4f}")

