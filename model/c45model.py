import pandas as pd
import numpy as np
from math import log2
from pprint import pprint

#Fungsi Utama C4.5

def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy_value = 0
    for i in range(len(elements)):
        p = counts[i] / np.sum(counts)
        entropy_value += -p * log2(p)
    return entropy_value

def InfoGain(data, split_attribute_name, target_name):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    Weighted_Entropy = 0
    for i in range(len(vals)):
        subset = data[data[split_attribute_name] == vals[i]]
        Weighted_Entropy += (counts[i]/np.sum(counts)) * entropy(subset[target_name])
    return total_entropy - Weighted_Entropy

def SplitInfo(data, split_attribute_name):
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    split_info = 0
    for i in range(len(vals)):
        p = counts[i]/np.sum(counts)
        split_info += -p * log2(p)
    return split_info

def GainRatio(data, split_attribute_name, target_name):
    IG = InfoGain(data, split_attribute_name, target_name)
    SI = SplitInfo(data, split_attribute_name)
    return IG / SI if SI != 0 else 0

def C45(data, originaldata, features, target_attribute_name, parent_node_class=None):
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]
    elif len(features) == 0:
        return parent_node_class
    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        item_values = [GainRatio(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        tree = {best_feature: {}}
        features = [i for i in features if i != best_feature]
        for value in np.unique(data[best_feature]):
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = C45(sub_data, data, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree
        return tree

#Fungsi Diskritisasi & Build Tree

def build_tree():
    data = pd.read_csv('Iris.csv')
    data = data.drop('Id', axis=1)
    for col in ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']:
        data[col] = pd.cut(data[col], bins=3, labels=["Low", "Medium", "High"])
    tree = C45(data, data, data.columns[:-1], 'Species')
    return tree

#Fungsi Prediksi

def predict(query, tree, default='Iris-setosa'):
    for key in list(query.keys()):
        if key in tree.keys():
            try:
                result = tree[key][query[key]]
            except:
                return default
            if isinstance(result, dict):
                return predict(query, result)
            else:
                return result
    return default

#Fungsi Konversi Kategori\

def kategori(nilai, mnLow, mxLow, mnMedium, mxMedium, mnHigh, mxHigh):
    if mnLow <= nilai <= mxLow:
        return 'Low'
    elif mnMedium <= nilai <= mxMedium:
        return 'Medium'
    elif mnHigh <= nilai <= mxHigh:
        return 'High'
    else:
        return None
