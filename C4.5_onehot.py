import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from graphviz import Source


# Function to calculate Entropy
def entropy(data):
    labels = data.iloc[:, 0] if isinstance(data, pd.DataFrame) else data
    probabilities = labels.value_counts(normalize=True)
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Avoid log(0)


# Information Gain and Gain Ratio calculation
def information_gain_ratio(data, feature):
    total_entropy = entropy(data)
    values = data[feature].unique()
    weighted_entropy = sum(
        (len(data[data[feature] == v]) / len(data)) * entropy(data[data[feature] == v])
        for v in values
    )
    info_gain = total_entropy - weighted_entropy
    split_info = entropy(data[feature])
    gain_ratio = info_gain / (split_info + 1e-9)  # Information Gain Ratio
    return gain_ratio


# Cross Validation using scikit-learn's DecisionTreeClassifier and custom feature selection
def cross_validate_C45(data, features):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []

    for train_index, test_index in kf.split(data):
        train_data, test_data = data.iloc[train_index], data.iloc[test_index]

        # Find the best feature based on gain ratio for training
        clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
        clf.fit(train_data[features], train_data.iloc[:, 0])

        predictions = clf.predict(test_data[features])
        accuracies.append(accuracy_score(test_data.iloc[:, 0], predictions))
    return np.mean(accuracies)


# Visualize Information Gain Ratio
def visualize_metrics_C45(data, features):
    gain_ratios = [information_gain_ratio(data, feature) for feature in features]
    x = np.arange(len(features))
    plt.figure(figsize=(12, 8))
    plt.bar(x, gain_ratios, width=0.4, label='Information Gain Ratio')
    plt.xlabel('Features')
    plt.ylabel('Score')
    plt.title('Feature Selection Metrics (Information Gain Ratio)')
    plt.xticks(x, features, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Visualize Tree using Graphviz (updated method)
def visualize_tree(clf, feature_names):
    dot_data = export_graphviz(
        clf,
        out_file=None,
        feature_names=feature_names,
        class_names=clf.classes_.astype(str),
        filled=True,
        rounded=True,
        special_characters=True
    )
    return Source(dot_data)


# Load the data
data = pd.read_csv('C:/Users\Timothy\PycharmProjects\pythonProject\machine learning/2\mushrooms.csv')

# Convert categorical features to one-hot encoded numerical features
data_encoded = pd.get_dummies(data, columns=data.columns[1:])  # One-hot encode all features except the label
features = data_encoded.columns[1:]  # Updated features after one-hot encoding

# Visualize Information Gain Ratio
visualize_metrics_C45(data, data.columns[1:])

# 5-Fold Cross Validation
accuracy_gain_ratio = cross_validate_C45(data_encoded, features)
print(f"Average accuracy with Information Gain Ratio (C4.5): {accuracy_gain_ratio:.4f}")

# Build the decision tree and visualize it
clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
clf.fit(data_encoded[features], data_encoded.iloc[:, 0])

# Generate the visualization with Graphviz and display it
dot = visualize_tree(clf, features)
dot.render("decision_tree_C45", format="png", cleanup=False)  # Save as PNG image

# Display the saved tree image inline
import matplotlib.image as mpimg

img = mpimg.imread("decision_tree_C45.png")
plt.figure(figsize=(12, 12))
plt.imshow(img)
plt.axis('off')
plt.show()
