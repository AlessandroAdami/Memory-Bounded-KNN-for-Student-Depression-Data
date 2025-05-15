from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

for k in range(1, 60, 20):
    classifier = KNeighborsClassifier(n_neighbors=k,weights='distance')

    # Fit the classifier to the training data
    classifier.fit(X_train, y_train)
    # Make predictions on the test data
    y_pred = classifier.predict(X_test)
    # Evaluate the classifier
    print(f"k={k}")
    print(f"Accuracy = {classifier.score(X_test, y_test)}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print('-' * 40)

