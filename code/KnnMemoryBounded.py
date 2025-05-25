from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from util import *
from concurrent.futures import ThreadPoolExecutor, as_completed # for parallel computation



class KNNMemoryBounded:
    """
    KNNMemoryBounded is a class that implements a memory-bounded version of the K-Nearest Neighbors algorithm.
    It uses a fixed-size memory buffer to store the training data and perform KNN classification.
    """

    def __init__(self, k=3, buffer_size=100, weights='distance', parallelize=True):
        """
        Initialize the KNNMemoryBounded classifier.

        Parameters:
        - k: Number of neighbors to consider for classification.
        - buffer_size: Maximum number of training samples to store in memory.
        - weights: function KNN uses to calculate distances between points
        - parallelize: Whether to fit the subdatasets in parallel or sequentially
        """
        self.k = k
        self.buffer_size = buffer_size
        self.weights = weights
        self.classifier = None
        self.parallelize = parallelize

    def fit(self, X, y, n_subdatasets=10):
        self.fit_parallel(X,y,n_subdatasets=n_subdatasets) if self.parallelize else self.fit_sequential(X,y,n_subdatasets=n_subdatasets)

    def fit_sequential(self, X, y, n_subdatasets):
        """
        Fit the KNNMemoryBounded classifier to the training data.

        Parameters:
        - X: Training data features.
        - y: Training data labels.
        - iterations: the number of random subdatasets we check
        """

        best_model = None
        best_f1_score = 0

        for _ in range(n_subdatasets):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.buffer_size)
            classifier = KNeighborsClassifier(n_neighbors=self.k, weights=self.weights)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            classification_dict = classification_report(y_test, y_pred, output_dict=True)

            f1_score = classification_dict['macro avg']['f1-score']
            if f1_score > best_f1_score:
                best_f1_score = f1_score
                best_model = classifier

        self.classifier = best_model
    
    def fit_parallel(self, X, y, n_subdatasets):
        """
        Fit the KNNMemoryBounded classifier to the training data.

        Parameters:
        - X: Training data features.
        - y: Training data labels.
        - iterations: the number of random subdatasets we check
        """
        
        def evaluate_subset(_):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.buffer_size)
            classifier = KNeighborsClassifier(n_neighbors=self.k, weights=self.weights)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            classification_dict = classification_report(y_test, y_pred, output_dict=True)
            f1 = classification_dict['macro avg']['f1-score']
            return f1, classifier

        best_model = None
        best_f1_score = 0

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(evaluate_subset, _) for _ in range(n_subdatasets)]
            for future in as_completed(futures):
                f1_score, model = future.result()
                if f1_score > best_f1_score:
                    best_f1_score = f1_score
                    best_model = model

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