import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder
from graphviz import Source


# Enhanced Gini Impurity calculation for categorical features
def gini_impurity_feature(data, feature):
    total_instances = len(data)
    gini = 0.0
    for category, subset in data.groupby(feature):
        subset_size = len(subset)
        subset_labels = subset.iloc[:, 0]
        subset_probabilities = subset_labels.value_counts(normalize=True)
        subset_gini = 1 - np.sum(subset_probabilities ** 2)
        gini += (subset_size / total_instances) * subset_gini
    return gini


# Cross Validation using pruned DecisionTreeClassifier with CART
def cross_validate_pruned_cart(data, features):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    for train_index, test_index in kf.split(data):
        train_data, test_data = data.iloc[train_index], data.iloc[test_index]

        # Train a pruned CART decision tree
        clf = DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_split=10, min_samples_leaf=5,
                                     random_state=42)
        clf.fit(train_data[features], train_data.iloc[:, 0])

        predictions = clf.predict(test_data[features])
        accuracies.append(accuracy_score(test_data.iloc[:, 0], predictions))
    return np.mean(accuracies)


# Visualize enhanced Gini Impurity per feature
def visualize_metrics_cart(data, features):
    gini_scores = [gini_impurity_feature(data, feature) for feature in features]
    x = np.arange(len(features))
    plt.figure(figsize=(12, 8))
    plt.bar(x, gini_scores, width=0.4, label='Gini Impurity')
    plt.xlabel('Features')
    plt.ylabel('Gini Score')
    plt.title('Feature Selection Metrics (Gini Impurity)')
    plt.xticks(x, features, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Visualize Tree using Graphviz
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
data = pd.read_csv('C:/Users/Timothy/PycharmProjects/pythonProject/machine learning/2/mushrooms.csv')

# Apply Label Encoding
encoder = LabelEncoder()
data_encoded = data.copy()
for col in data.columns[1:]:
    data_encoded[col] = encoder.fit_transform(data[col])

features = data_encoded.columns[1:]

# Visualize Gini Impurity for each feature
visualize_metrics_cart(data_encoded, features)

# 5-Fold Cross Validation with pruning parameters
accuracy_pruned_gini = cross_validate_pruned_cart(data_encoded, features)
print(f"Average accuracy with pruned CART (Gini Impurity): {accuracy_pruned_gini:.4f}")

# Build and visualize a pruned decision tree using CART
clf_pruned = DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_split=10, min_samples_leaf=5,
                                    random_state=42)
clf_pruned.fit(data_encoded[features], data_encoded.iloc[:, 0])

# Generate the visualization with Graphviz and display it
dot_pruned = visualize_tree(clf_pruned, features)
dot_pruned.render("decision_tree_pruned_cart", format="png", cleanup=False)

# Display the saved pruned tree image inline
import matplotlib.image as mpimg

img_pruned = mpimg.imread("decision_tree_pruned_cart.png")
plt.figure(figsize=(12, 12))
plt.imshow(img_pruned)
plt.axis('off')
plt.show()
