from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


class KNNMemoryBounded:
    """
    KNNMemoryBounded is a class that implements a memory-bounded version of the K-Nearest Neighbors algorithm.
    It uses a fixed-size memory buffer to store the training data and performs KNN classification on the fly.
    """

    def __init__(self, k=3, buffer_size=100, weights='distance'):
        """
        Initialize the KNNMemoryBounded classifier.

        Parameters:
        - k: Number of neighbors to consider for classification.
        - buffer_size: Maximum number of training samples to store in memory.
        """
        self.k = k
        self.buffer_size = buffer_size
        self.weights = weights
        self.memory_bounded_classifier = None


    def fit(self, X, y, iterations=10):
        """
        Fit the KNNMemoryBounded classifier to the training data.

        Parameters:
        - X: Training data features.
        - y: Training data labels.
        """

        best_model = None
        best_accuracy = 0

        # Note: this can be done in parallel but for simplicity we will do it in a loop
        for _ in range(iterations):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.buffer_size)
            classifier = KNeighborsClassifier(n_neighbors=self.k, weights=self.weights)
            classifier.fit(X_train, y_train)
            y_pred = self.classifier.predict(X_test)
            accuracy = self.classifier.score(X_test, y_test)
            confusion_matrix = self.confusionMatrix(y_test, y_pred)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = self.classifier

            # TODO: think of a way to pick the best model based on accuracy and confusion matrix

        self.memory_bounded_classifier = best_model

    def predict(self, X):
        """
        Predict the class labels for the input data.

        Parameters:
        - X: Input data features.

        Returns:
        - Predicted class labels.
        """
        if self.memory_bounded_classifier is None:
            raise ValueError("The model has not been fitted yet. Please call fit() before predict().")
        return self.memory_bounded_classifier.predict(X)
        