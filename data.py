import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# 计算信息熵
def entropy(data):
    label_counts = data.value_counts(normalize=True)
    return -np.sum(label_counts * np.log2(label_counts))


# 计算条件熵
def conditional_entropy(data, feature):
    total_instances = len(data)
    feature_values = data[feature].unique()
    cond_entropy = 0.0
    for value in feature_values:
        subset = data[data[feature] == value]
        prob = len(subset) / total_instances
        cond_entropy += prob * entropy(subset.iloc[:, 0])  # 使用标签列计算熵
    return cond_entropy


# 计算信息增益
def information_gain(data, feature):
    return entropy(data.iloc[:, 0]) - conditional_entropy(data, feature)


# 计算信息增益率
def gain_ratio(data, feature):
    feature_entropy = entropy(data[feature])
    if feature_entropy == 0:
        return 0  # 如果特征熵为0，信息增益率为0
    return information_gain(data, feature) / feature_entropy


# 计算基尼系数
def gini_impurity(data, feature):
    total_instances = len(data)
    gini = 0.0
    for category, subset in data.groupby(feature):
        subset_size = len(subset)
        subset_labels = subset.iloc[:, 0]
        subset_probabilities = subset_labels.value_counts(normalize=True)
        subset_gini = 1 - np.sum(subset_probabilities ** 2)
        gini += (subset_size / total_instances) * subset_gini
    return gini


# 主程序
def calculate_metrics(data):
    metrics = {
        'Feature': [],
        'Entropy': [],
        'Information Gain': [],
        'Gain Ratio': [],
        'Gini Impurity': []
    }

    for feature in data.columns[1:]:  # 遍历所有特征列（假设标签列是第0列）
        metrics['Feature'].append(feature)
        metrics['Entropy'].append(entropy(data[feature]))
        metrics['Information Gain'].append(information_gain(data, feature))
        metrics['Gain Ratio'].append(gain_ratio(data, feature))
        metrics['Gini Impurity'].append(gini_impurity(data, feature))

    return pd.DataFrame(metrics)


# 可视化各特征的指标
def visualize_metrics(metrics_df):
    x = np.arange(len(metrics_df))

    plt.figure(figsize=(14, 8))
    plt.bar(x - 0.2, metrics_df['Entropy'], width=0.2, label='Entropy')
    plt.bar(x, metrics_df['Information Gain'], width=0.2, label='Information Gain')
    plt.bar(x + 0.2, metrics_df['Gain Ratio'], width=0.2, label='Gain Ratio')
    plt.bar(x + 0.4, metrics_df['Gini Impurity'], width=0.2, label='Gini Impurity')

    plt.xlabel('Features')
    plt.ylabel('Scores')
    plt.title('Feature Metrics: Entropy, Information Gain, Gain Ratio, Gini Impurity')
    plt.xticks(x, metrics_df['Feature'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


# 加载数据并进行标签编码
data = pd.read_csv('C:/Users/Timothy/PycharmProjects/pythonProject/machine learning/2/mushrooms.csv')

# 标签编码
encoder = LabelEncoder()
data_encoded = data.copy()
for col in data.columns[1:]:
    data_encoded[col] = encoder.fit_transform(data[col])

# 计算各特征的指标
metrics_df = calculate_metrics(data_encoded)

# 可视化结果
visualize_metrics(metrics_df)
