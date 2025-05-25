from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from knnMemoryBounded import KNNMemoryBounded

import pandas as pd
import matplotlib.pyplot as plt
from util import *
from time import time

"""
This script compares regular KNN with KNNMemoryBounded for the given dataset.
It saves some relevant plots in the "./plots" folder.
Note: when using this script on another dataset make sure that its data is numerical for KNN distance computations.
"""

start_time = time()

df = pd.read_csv('../data/reduced_student_depression_dataset.csv')
X = df.drop(columns=['Depression'])
y = df['Depression']

print(X[0:5])

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Store metrics for plotting
k_values = [1,2,4,8,16]
metrics = {
    'k': [],
    'accuracy_regular': [],
    'precision_regular': [],
    'recall_regular': [],
    'f1_regular': [],
    'accuracy_memory': [],
    'precision_memory': [],
    'recall_memory': [],
    'f1_memory': []
}

for k in k_values:
    age = df['Age']
    print(min(age))
    print(max(age))
    classifier = KNeighborsClassifier(n_neighbors=k, weights='distance')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print(f"k={k}")
    acc = classifier.score(X_test, y_test)
    print(f"Accuracy = {acc}")
    print(confusionMatrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)
    f1 = report['macro avg']['f1-score']
    prec = report['macro avg']['precision']
    rec = report['macro avg']['recall']
    print(f"F1 score = {f1}")
    print('%' * 40)

    # Store regular metrics
    metrics['k'].append(k)
    metrics['accuracy_regular'].append(acc)
    metrics['precision_regular'].append(prec)
    metrics['recall_regular'].append(rec)
    metrics['f1_regular'].append(f1)

    print("Memory bounded classifier:")
    classifier = KNNMemoryBounded(k=k, buffer_size=500, weights='distance',parallelize=True) # less than 2% of all the points!
    classifier.fit(X_train, y_train,n_subdatasets=10) # 10 random KNN models checked
    y_pred = classifier.predict(X_test)
    acc = classifier.score(X_test, y_test)
    print(f"Accuracy = {acc}")
    print(confusionMatrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)
    f1 = report['macro avg']['f1-score']
    prec = report['macro avg']['precision']
    rec = report['macro avg']['recall']
    print(f"F1 score = {f1}")
    print('-' * 40)

    # Store memory-bounded metrics
    metrics['accuracy_memory'].append(acc)
    metrics['precision_memory'].append(prec)
    metrics['recall_memory'].append(rec)
    metrics['f1_memory'].append(f1)

# Plot results

bar_width = 0.35
x = range(len(metrics['k']))

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
titles = ['Accuracy', 'Precision', 'Recall', 'F1-score']
metric_keys = [
    ('accuracy_regular', 'accuracy_memory'),
    ('precision_regular', 'precision_memory'),
    ('recall_regular', 'recall_memory'),
    ('f1_regular', 'f1_memory')
]

for ax, title, keys in zip(axs.ravel(), titles, metric_keys):
    ax.bar([i - bar_width/2 for i in x], metrics[keys[0]], width=bar_width, label='Regular KNN')
    ax.bar([i + bar_width/2 for i in x], metrics[keys[1]], width=bar_width, label='Memory-Bounded KNN')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics['k'])
    ax.set_xlabel('k')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True)

plt.suptitle('Comparison of KNN vs Memory-Bounded KNN')
plt.tight_layout()
plt.savefig('plots/knn_comparison.png', dpi=300)
#plt.show()
print(time() - start_time)