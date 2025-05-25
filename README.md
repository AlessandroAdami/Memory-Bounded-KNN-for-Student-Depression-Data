## Memory Bounded K-Nearest Neighbours for Large Student Depression Dataset

### About

This work was done during the Summer Reaserch Experiences program at the University of British Columbia Mathematics department. The topic of interest for this reaserch is the K-Nearest Neighbours (KNN) algorithm. More specifically the goal was to try and overcome the main challenges KNN faces when used with datasets with a large number of points $n$. When $n$ is large not only does KNN encounter some memory issue, but it becomes solwer in its predictions.

### The KNN Algorithm

KNN is a supervised learning algorithm with one hyperparameter $k$. It's exectution for predictions can be explained as follows:
- Store all examples and their labels from the training set in memory.
- To predict the label of an unseen example $x$:
    - Calculate the distances between $x$ and every training example.
    - Find the $k$-nearest neighbours of $x$.
    - Give $x$ the most common label among its nearest neighbours.

![Demo Animation](knn_visualization/media/gifs/Knn.gif)

### Memory Bounded KNN

The KNN algorithm requires $O(nd)$ memory usage, where $n$ is the number of training examples and $d$ is the number of features of each example. Predicting the label of one unseen example takes $O(nd)$ time, as we need to calculate one distance for every training example.

The idea between Memory Bounded KNN (MBKNN) is to take a subset of the training dataset with fixed size $m$, and to only use those points in our predictions. This brings our memory usage and prediction time down to $O(d)$.

![Demo Animation Two](knn_visualization/media/gifs/KnnMemoryBounded.gif)