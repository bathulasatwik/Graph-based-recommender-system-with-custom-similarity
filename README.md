

# Collaborative Filtering with Graph Convolutional Networks (GCN) with custom similarity on the MovieLens Dataset

## Table of Contents
- [Overview](#overview)
- [Data Preprocessing](#data-preprocessing)
- [Custom Similarity Metric](#custom-similarity-metric)
- [Graph Construction](#graph-construction)
- [GCN Architecture](#gcn-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Predictions](#predictions)
- [Top K Precision](#top-k-precision)
- [Dependencies](#dependencies)
  

## Overview
This repository implements a collaborative filtering approach using a Graph Convolutional Network (GCN) on the MovieLens 100k dataset. The primary goal is to predict user ratings for movies by modeling user-item interactions as a graph. A **custom similarity metric** is employed to compute user similarities, enhancing the model's precision by integrating mathematically robust and stable transformations. The GCN architecture is then used to propagate this information and generate accurate recommendations.

## Data Preprocessing
The MovieLens 100k dataset is first loaded and processed to create a user-item matrix:
- The dataset is read using `pandas`, and the timestamp information is dropped.
- Missing values in the user-item matrix are filled with zeros to prepare the data for further processing.

```python
url = 'http://files.grouplens.org/datasets/movielens/ml-100k/u.data'
columns = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv(url, sep='\t', names=columns)
df.drop('timestamp', axis=1, inplace=True)
user_item_matrix = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
```

## Custom Similarity Metric
A custom similarity metric is designed to calculate the similarity between users in a stable and sensitive manner:

1. Stable Softmax Function: Converts each user’s rating vector into a probability distribution to prevent numerical instability.

$$
\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}
$$
2. Exponential Transformation: Weighs the transformed vector entries using an exponential function of their indices, highlighting higher preferences.
$$
v_{\text{exp}} = v \cdot e^{v \cdot \text{indices}(v)}
$$

3. Entropy Calculation: Entropies of the transformed vectors are computed, and these values are combined to determine similarity scores.

$$
\text{Similarity} = e^{H(v_1)} + e^{H(v_2)}
$$




```python
def stable_softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def custom_similarity(v1, v2):
    epsilon = 1e-8  # Small value to avoid division by zero
    v1 = stable_softmax(v1)
    v2 = stable_softmax(v2)

    if np.isnan(v1).any() or np.isnan(v2).any():
        print("NaN detected in softmax output")
        return np.nan

    indices_v1 = np.arange(1, len(v1) + 1)
    v1_exp = v1 * np.exp(v1 * indices_v1)

    indices_v2 = np.arange(1, len(v2) + 1)
    v2_exp = v2 * np.exp(v2 * indices_v2)

    if np.isnan(v1_exp).any() or np.isnan(v2_exp).any():
        print("NaN detected in exponential transformation")
        return np.nan

    joint_distribution = np.outer(v1_exp, v2_exp)
    v1_flat = v1_exp.flatten()
    v2_flat = v2_exp.flatten()
    joint_flat = joint_distribution.flatten()

    H_v1 = entropy(v1_flat + epsilon)  # Add epsilon to avoid log(0)
    H_v2 = entropy(v2_flat + epsilon)  # Add epsilon to avoid log(0)

    def sum_exponentials_metric(a, b):
        return np.exp(a) + np.exp(b)

    return sum_exponentials_metric(H_v1, H_v2)

```

## Graph Construction

After computing the user similarity matrix, a graph is formed where:

- Nodes represent users.
- Edges are created between users based on their similarity scores.
- The graph structure allows the GCN to propagate and aggregate user-item information effectively.

```python
edge_index = []
edge_weight = []

for i in range(num_users):
    for j in range(num_users):
        if i != j:
            edge_index.append([i, j])
            edge_weight.append(similarity_matrix[i, j])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
edge_weight = torch.tensor(edge_weight, dtype=torch.float)
```

## GCN Architecture 

We define a GCN architecture consisting of four layers:

- Input Layer: Takes the user-item matrix as input.
- Hidden Layers: Three hidden layers with dimensions 64, 32, and 16, each followed by a ReLU activation function.
- Output Layer: Produces the final user representation used for predictions.

```python

class DeeperWeightedGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeeperWeightedGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 64, normalize=True, add_self_loops=False)
        self.conv2 = GCNConv(64, 32, normalize=True, add_self_loops=False)
        self.conv3 = GCNConv(32, 16, normalize=True, add_self_loops=False)
        self.conv4 = GCNConv(16, out_channels, normalize=True, add_self_loops=False)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv4(x, edge_index, edge_weight=edge_weight)
        return x

```

## Training and Evaluation

The model is trained using Mean Squared Error (MSE) loss, with values clipped to the valid rating range (1 to 5). Regular checks are performed during training to monitor stability and identify any NaN values.

```python

for epoch in range(500):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    loss = criterion(out[data.x != 0], data.x[data.x != 0])
    loss.backward()
    optimizer.step()

```

## Predictions

Once the model is trained, it predicts ratings for unrated items. These predictions are integrated into the user-item matrix. The values are constrained to ensure they remain between 1 and 5.


```python
# Predicting the ratings
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index, data.edge_attr)

# Applying the constraint: clip the values between 1 and 5
out = torch.clamp(out, min=1.0, max=5.0)

predicted_ratings = pd.DataFrame(out.numpy(), index=user_item_matrix.index, columns=user_item_matrix.columns)

for user in user_item_matrix.index:
    for movie in user_item_matrix.columns:
        if user_item_matrix.at[user, movie] == 0:
            user_item_matrix.at[user, movie] = predicted_ratings.at[user, movie]

```

## Top K Precision

```python
def precision_at_k(predictions, actuals, k):
    top_k_pred = np.argsort(predictions)[-k:]
    relevant_items = np.nonzero(actuals)[0]
    precision = len(set(top_k_pred).intersection(set(relevant_items))) / k
    return precision

k = 10
num_users = before.shape[0]  
total_precision = 0

for user_id in range(num_users):
    precision = precision_at_k(after[user_id], before[user_id], k)
    total_precision += precision

average_precision = total_precision / num_users

```


## Dependencies

To run this implementation, the following dependencies are required:

- `pandas` and `numpy` for data manipulation and numerical operations.
- `torch` and `torch_geometric` for model implementation and graph processing.
- `scipy` for computing the custom similarity metric.

