from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


# Load the dataset

df = pd.read_csv('../data/cleaned_student_depression_dataset.csv')
# Drop the target column from the features
X = df.drop(columns=['Depression'])
y = df['Depression']


# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

for k in range(1, 100, 20):
    classifier = KNeighborsClassifier(n_neighbors=k,weights='distance')

    # Fit the classifier to the training data
    classifier.fit(X_train, y_train)
    # Make predictions on the test data
    y_pred = classifier.predict(X_test)
    # Evaluate the classifier
    print(classifier.score(X_test, y_test))

