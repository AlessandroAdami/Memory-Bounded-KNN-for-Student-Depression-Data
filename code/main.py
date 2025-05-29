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
    'cm_regular': [],
    'accuracy_memory': [],
    'precision_memory': [],
    'recall_memory': [],
    'f1_memory': [],
    'cm_memory': []
}

for k in k_values:
    classifier = KNeighborsClassifier(n_neighbors=k, weights='distance')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print(f"k={k}")
    acc = classifier.score(X_test, y_test)
    print(f"Accuracy = {acc}")
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
    metrics['cm_regular'].append(confusionMatrix(y_test, y_pred))

    print("Memory bounded classifier:")
    classifier = KNNMemoryBounded(k=k, buffer_size=500, weights='distance',parallelize=True) # less than 2% of all the points!
    classifier.fit(X_train, y_train,n_subdatasets=10) # 10 random KNN models checked
    y_pred = classifier.predict(X_test)
    acc = classifier.score(X_test, y_test)
    print(f"Accuracy = {acc}")
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
    metrics['cm_memory'].append(confusionMatrix(y_test, y_pred))

# Plot accuracy results

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

plt.suptitle('Comparison of KNN vs Memory-Bounded KNN',fontsize=18,fontweight='bold')
plt.tight_layout()
plt.savefig('plots/knn_comparison.png', dpi=300)

# Plot heatmaps for confusion matrices

for i, k in enumerate(metrics['k']):
    cm_reg = metrics['cm_regular'][i]
    cm_mem = metrics['cm_memory'][i]

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f'Confusion Matrix Comparison for k={k}', fontsize=16)

    # --- Regular KNN ---
    im1 = axs[0].imshow(cm_reg, cmap='Oranges')
    axs[0].set_title("Regular KNN")
    axs[0].set_xlabel("Predicted")
    axs[0].set_ylabel("True")
    axs[0].set_xticks([0, 1])
    axs[0].set_yticks([0, 1])
    axs[0].set_xticklabels(['1', '0'])
    axs[0].set_yticklabels(['1', '0'])

    for (x, y), val in np.ndenumerate(cm_reg):
        text_color = "white" if cm_reg[x, y] > cm_reg.max() / 2 else "black"
        axs[0].text(y, x, f"{val}", ha="center", va="center", color=text_color)

    axs[0].set_xticks(np.arange(-0.5, 2, 1), minor=True)
    axs[0].set_yticks(np.arange(-0.5, 2, 1), minor=True)
    axs[0].grid(which="minor", color="gray", linestyle="--", linewidth=0.5)
    axs[0].tick_params(which="minor", bottom=False, left=False)

    # --- Memory-Bounded KNN ---
    im2 = axs[1].imshow(cm_mem, cmap='Oranges')
    axs[1].set_title("Memory-Bounded KNN")
    axs[1].set_xlabel("Predicted")
    axs[1].set_ylabel("True")
    axs[1].set_xticks([0, 1])
    axs[1].set_yticks([0, 1])
    axs[1].set_xticklabels(['1', '0'])
    axs[1].set_yticklabels(['1', '0'])

    for (x, y), val in np.ndenumerate(cm_mem):
        text_color = "white" if cm_mem[x, y] > cm_mem.max() / 2 else "black"
        axs[1].text(y, x, f"{val}", ha="center", va="center", color=text_color)

    axs[1].set_xticks(np.arange(-0.5, 2, 1), minor=True)
    axs[1].set_yticks(np.arange(-0.5, 2, 1), minor=True)
    axs[1].grid(which="minor", color="gray", linestyle="--", linewidth=0.5)
    axs[1].tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'plots/confusion_matrices/conf_matrix_k={k}.png', dpi=300)
    plt.close()

print(time() - start_time)