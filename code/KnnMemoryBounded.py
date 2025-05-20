from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from util import *


class KNNMemoryBounded:
    """
    KNNMemoryBounded is a class that implements a memory-bounded version of the K-Nearest Neighbors algorithm.
    It uses a fixed-size memory buffer to store the training data and perform KNN classification.
    """

    def __init__(self, k=3, buffer_size=100, weights='distance'):
        """
        Initialize the KNNMemoryBounded classifier.

        Parameters:
        - k: Number of neighbors to consider for classification.
        - buffer_size: Maximum number of training samples to store in memory.
        - weights: function KNN uses to calculate distances between points
        """
        self.k = k
        self.buffer_size = buffer_size
        self.weights = weights
        self.classifier = None


    def fit(self, X, y, iterations=10):
        """
        Fit the KNNMemoryBounded classifier to the training data.

        Parameters:
        - X: Training data features.
        - y: Training data labels.
        - iterations: the number of random subdatasets we check
        """

        best_model = None
        best_f1_score = 0

        # Note: this can be done in parallel, look into it
        for _ in range(iterations):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.buffer_size)
            classifier = KNeighborsClassifier(n_neighbors=self.k, weights=self.weights)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            # TODO: think of a way to pick the best model, for now we just use f1-score
            classification_dict = classification_report(y_test, y_pred, output_dict=True)
            f1_score = classification_dict['1.0']['f1-score']
            if f1_score > best_f1_score:
                best_f1_score = f1_score
                best_model = classifier

        self.classifier = best_model

    def predict(self, X):
        """
        Predict the class labels for the input data.

        Parameters:
        - X: Input data features.

        Returns:
        - Predicted class labels.
        """
        if self.classifier is None:
            raise ValueError("The model has not been fitted yet. Please call fit() before predict().")
        return self.classifier.predict(X)  

    def score(self, X, y):
        """
        Evaluate the classifier on the test data.

        Parameters:
        - X: Test data features.
        - y: Test data labels.

        Returns:
        - Accuracy score of the classifier.
        """
        if self.classifier is None:
            raise ValueError("The model has not been fitted yet. Please call fit() before score().")
        return self.classifier.score(X, y)      