import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('../data/cleaned_student_depression_dataset.csv')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = KNeighborsClassifier(n_neighbors=8) # k = 8 does well on the full dataset
sfs = SequentialFeatureSelector(knn, n_features_to_select=8, direction='forward') # fiddle around w/ n_features
sfs.fit(X_scaled, y)
selected_features = sfs.get_support(indices=True)
X_selected = X.iloc[:, selected_features]
reduced_df = pd.concat([X_selected, y.reset_index(drop=True)], axis=1)
reduced_df.to_csv('../data/reduced_student_depression_dataset.csv', index=False)