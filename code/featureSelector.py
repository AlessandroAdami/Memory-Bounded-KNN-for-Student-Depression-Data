import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import StandardScaler
"""
This script does feature selection on the *cleaned* dataset. We use greedy forward selection by default.
The cleaned dataset, converted to a numerical space for KNN, has eighteen features. 
Using n_features_to_select = 8 already gets significant results
"""


df = pd.read_csv('../data/cleaned_student_depression_dataset.csv')
# df = df.drop(columns='Have you ever had suicidal thoughts ?') # dropping this column causes a significant loss in accuracy

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Scale the dataset so that each feature contribues fairly to the distance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = KNeighborsClassifier(n_neighbors=8) # k = 8 works well for the student depression dataset
sfs = SequentialFeatureSelector(knn, n_features_to_select=5, direction='forward') # n_features_to_select = 5 works well
sfs.fit(X_scaled, y)
selected_features = sfs.get_support(indices=True)
X_selected = X.iloc[:, selected_features]
reduced_df = pd.concat([X_selected, y.reset_index(drop=True)], axis=1)
reduced_df.to_csv('../data/reduced_student_depression_dataset.csv', index=False)