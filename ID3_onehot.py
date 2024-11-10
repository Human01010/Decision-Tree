import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from graphviz import Digraph

# Function to calculate Entropy
def entropy(data):
    labels = data.iloc[:, 0] if isinstance(data, pd.DataFrame) else data
    probabilities = labels.value_counts(normalize=True)
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Avoid log(0)

# Information gain calculation
def information_gain(data, feature):
    total_entropy = entropy(data)
    values = data[feature].unique()
    weighted_entropy = sum((len(data[data[feature] == v]) / len(data)) * entropy(data[data[feature] == v]) for v in values)
    info_gain = total_entropy - weighted_entropy
    info_gain_ratio = info_gain / (entropy(data[feature]) + 1e-9)  # Information Gain Ratio
    return info_gain, info_gain_ratio

# ID3 algorithm
def id3(data, features, depth=0):
    labels = data.iloc[:, 0]
    if labels.nunique() == 1:
        return labels.iloc[0]
    if len(features) == 0:
        return labels.mode()[0]
    gains = [information_gain(data, feature)[0] for feature in features]
    best_feature = features[np.argmax(gains)]
    tree = {best_feature: {}}
    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        subtree = id3(subset, features[features != best_feature], depth + 1)
        tree[best_feature][value] = subtree
    return tree

# Predict function
def predict(tree, instance):
    if not isinstance(tree, dict):
        return tree
    feature = next(iter(tree))
    value = instance[feature]
    subtree = tree[feature].get(value, None)
    return predict(subtree, instance) if subtree else None

# Cross Validation
def cross_validate(data, features):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    for train_index, test_index in kf.split(data):
        train_data, test_data = data.iloc[train_index], data.iloc[test_index]
        tree = id3(train_data, features)
        predictions = [predict(tree, row) for _, row in test_data.iterrows()]
        accuracies.append(accuracy_score(test_data.iloc[:, 0], predictions))
    return np.mean(accuracies)

# Visualize Information Gain
def visualize_metrics(data, features):
    gains, gain_ratios = zip(*[information_gain(data, feature) for feature in features])
    x = np.arange(len(features))
    plt.figure(figsize=(12, 8))
    plt.bar(x - 0.2, gains, width=0.4, label='Information Gain')
    plt.bar(x + 0.2, gain_ratios, width=0.4, label='Information Gain Ratio')
    plt.xlabel('Features')
    plt.ylabel('Score')
    plt.title('Feature Selection Metrics Comparison')
    plt.xticks(x, features, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Visualize Tree using Graphviz
def visualize_tree(tree, dot=None, parent_name=None, edge_label=""):
    if dot is None:
        dot = Digraph()
    if isinstance(tree, dict):
        feature = next(iter(tree))
        node_name = f"{feature}"
        dot.node(node_name, feature)
        if parent_name:
            dot.edge(parent_name, node_name, label=edge_label)
        for value, subtree in tree[feature].items():
            visualize_tree(subtree, dot, node_name, str(value))
    else:
        leaf_name = f"{tree}"
        dot.node(leaf_name, str(tree), shape="ellipse")
        dot.edge(parent_name, leaf_name, label=edge_label)
    return dot

# Load the data
data = pd.read_csv('C:/Users\Timothy\PycharmProjects\pythonProject\machine learning/2\mushrooms.csv')
features = data.columns[1:]  # Exclude label column

# Visualize Information Gain and Gain Ratio
visualize_metrics(data, features)

# 5-Fold Cross Validation
accuracy_entropy = cross_validate(data, features)
print(f"Average accuracy with Information Gain: {accuracy_entropy:.4f}")

# Build the decision tree and visualize it
tree = id3(data, features)
dot = visualize_tree(tree)
dot.render("decision_tree", format="png", cleanup=False)  # Save as PNG image

# Visualize tree rendering and display it (this part is environment-dependent)
plt.imshow(plt.imread("decision_tree.png"))
plt.axis('off')
plt.show()
