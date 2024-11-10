import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder
from graphviz import Source


# Function to calculate Entropy
def entropy(data):
    labels = data.iloc[:, 0] if isinstance(data, pd.DataFrame) else data
    probabilities = labels.value_counts(normalize=True)
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Avoid log(0)


# Information gain calculation
def information_gain(data, feature):
    total_entropy = entropy(data)
    values = data[feature].unique()
    weighted_entropy = sum(
        (len(data[data[feature] == v]) / len(data)) * entropy(data[data[feature] == v]) for v in values)
    info_gain = total_entropy - weighted_entropy
    info_gain_ratio = info_gain / (entropy(data[feature]) + 1e-9)  # Information Gain Ratio
    return info_gain, info_gain_ratio


# Cross Validation using scikit-learn's DecisionTreeClassifier
def cross_validate(data, features):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    for train_index, test_index in kf.split(data):
        train_data, test_data = data.iloc[train_index], data.iloc[test_index]

        # Training using scikit-learn's DecisionTreeClassifier with 'entropy' criterion
        clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
        clf.fit(train_data[features], train_data.iloc[:, 0])

        predictions = clf.predict(test_data[features])
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
data = pd.read_csv('C:/Users/Timothy/PycharmProjects/pythonProject/machine learning/2/mushrooms.csv')

# Convert categorical features to Label Encoded numerical features
label_encoder = LabelEncoder()

# Apply label encoding to each feature except the label (first column)
for col in data.columns:
    data[col] = label_encoder.fit_transform(data[col])

features = data.columns[1:]  # Updated features after label encoding

# Visualize Information Gain and Gain Ratio
visualize_metrics(data, data.columns[1:])

# 5-Fold Cross Validation
accuracy_entropy = cross_validate(data, features)
print(f"Average accuracy with Information Gain: {accuracy_entropy:.4f}")

# Build the decision tree and visualize it
clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
clf.fit(data[features], data.iloc[:, 0])

# Generate the visualization with Graphviz and display it
dot = visualize_tree(clf, features)
dot.render("decision_tree", format="png", cleanup=False)  # Save as PNG image

# Open the PNG image in the default image viewer
dot.view("decision_tree")

# Display the saved tree image inline
import matplotlib.image as mpimg

# Display the saved tree image inline with matplotlib
img = mpimg.imread("decision_tree.png")
plt.figure(figsize=(12, 12))
plt.imshow(img)
plt.axis('off')
plt.show()
