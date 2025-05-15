from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np


# Load the dataset

df = pd.read_csv('../data/cleaned_student_depression_dataset.csv')
# Drop the target column from the features
X = df.drop(columns=['Depression'])
y = df['Depression']

# Sandardize the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def confusionMatrix(y_test, y_pred):
    """
    Returns confusion matrix accoring to standard format
    """
    cm = confusion_matrix(y_test, y_pred)
    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    return np.array([[TP, FN],
                     [FP, TN]])

for k in [1,20,40,60]:
    classifier = KNeighborsClassifier(n_neighbors=k,weights='distance')

    # Fit the classifier to the training data and make predictions
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    # Evaluate the classifier
    print(f"k={k}")
    print(f"Accuracy = {classifier.score(X_test, y_test)}")
    print(confusionMatrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print('-' * 40)