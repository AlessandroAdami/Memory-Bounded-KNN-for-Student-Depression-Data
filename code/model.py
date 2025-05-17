from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from knnMemoryBounded import KNNMemoryBounded
import pandas as pd
import numpy as np
from util import *


# Load dataset

df = pd.read_csv('../data/cleaned_student_depression_dataset.csv')
# Drop target column from features
X = df.drop(columns=['Depression'])
y = df['Depression']

print(X[0:5])

# Sandardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Split dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



for k in [1,20,40,60]:
    classifier = KNeighborsClassifier(n_neighbors=k,weights='distance')

    # Fit classifier to training data and make predictions
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    # Evaluate classifier
    print(f"k={k}")
    print(f"Accuracy = {classifier.score(X_test, y_test)}")
    print(confusionMatrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    dict = classification_report(y_test, y_pred, output_dict=True)
    print(f"F1 score = {dict['1.0']['f1-score']}")

    print('%' * 40)

    print("Memory bounded classifier:")
    classifier = KNNMemoryBounded(k=k, buffer_size=500, weights='distance')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    # Evaluate classifier
    print(f"Accuracy = {classifier.score(X_test, y_test)}")
    print(confusionMatrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    dict = classification_report(y_test, y_pred, output_dict=True)
    print(f"F1 score = {dict['1.0']['f1-score']}")
    print('-' * 40)