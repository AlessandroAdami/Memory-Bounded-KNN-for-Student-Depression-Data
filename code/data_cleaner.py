import pandas as pd
import numpy as np
# Packages for imputation
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

"""
This script is used to clean the dataset for the student depression prediction model.
The dataset is expected to be in CSV format and should contain the following columns:
- id: Unique identifier for each entry
...

We encode Male as 0 and Female as 1.
"""
df = pd.read_csv('../data/student_depression_dataset.csv') # cd in the 'code' directory to use this

# Checking for duplicated ids
duplicate_count = df.duplicated(subset='id').sum()
if duplicate_count > 0:
    df.drop_duplicates(subset='id', inplace=True)

# Dropping unnecessary columns
df.drop(columns=["id"], inplace=True)
df.drop(columns=['City'], inplace=True)
df.drop(columns=['Profession'], inplace=True) # This column is filled with "Student"
df.drop(columns=['Work Pressure'], inplace=True) # This column is filled with 0.0s
df.drop(columns=['Job Satisfaction'], inplace=True) # This column is filled with 0.0s
#df.drop(columns=['Degree'], inplace=True)
 
for column in df.columns:
    print(f"Unique values in '{column}':")
    print(df[column].unique())
    print('-' * 40)

# Check for missing values, the dataset uses the value 'Others' to indicate missing values
# except for 'Finintial Stress' column which uses '?'

df.replace('Others', np.nan, inplace=True)
df.replace('?', np.nan, inplace=True)
missing_values = df.isnull().sum()
print("Missing values in each column:")
print(missing_values[missing_values > 0])


# Convert categorical columns to numerical values
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Family History of Mental Illness'] = df['Family History of Mental Illness'].map({'No': 0, 'Yes': 1})
df['Have you ever had suicidal thoughts ?'] = df['Have you ever had suicidal thoughts ?'].map({'No': 0, 'Yes': 1})

# Replace intial Stress with numerical values that are mean of the range
df['Sleep Duration'] = df['Sleep Duration'].map({'\'Less than 5 hours\'': 2.5, 
                                                 '\'5-6 hours\'': 5.5, 
                                                 '\'7-8 hours\'': 7.5, 
                                                 '\'More than 8 hours\'': 9.5})

# Replace 'Dieatary Habits' with one-hot encoding
one_hot = pd.get_dummies(df['Dietary Habits'])
one_hot.columns = ['Dietary_Habits_Unhealthy', 'Dietary_Habits_Moderate', 'Dietary_Habits_Healthy']
for label in one_hot.columns:
    one_hot[label] = one_hot[label].astype(int)
df = df.drop('Dietary Habits', axis=1)
df = pd.concat([df, one_hot], axis=1)

# Replace 'Degree' with one-hot encoding
df['Degree'] = df['Degree'].map({'B.Pharm': 0, 'BSc': 0, 'BA': 0, 'BCA': 0, 'B.Ed': 0, 'BE': 0, 'BHM': 0, 'B.Com': 0, 'B.Arch': 0,'B.Tech': 0, 'LLB': 0, 'MBBS': 0, 'BBA': 0,
                                 'M.Tech': 1, 'M.Ed': 1, 'MSc': 1, 'M.Pharm': 1, 'MCA': 1, 'MA': 1, 'MBA': 1, 'M.Com': 1, 'LLM': 1, 'ME': 1, 'MHM': 1,
                                 'PhD': 2, 'MD': 2,
                                 '\'Class 12\'': 3})
# Impute missing values using IterativeImputer with Regularized Liner Regression
imputer = IterativeImputer(max_iter=10, random_state=0)
df_imputed = imputer.fit_transform(df)
df = pd.DataFrame(df_imputed, columns=df.columns)
# Check for any remaining missing values
missing_values = df.isnull().sum()
print("Missing values in each column after imputation:" + str(missing_values.sum()))
print(missing_values[missing_values > 0])

# One-hot encoding for 'Degree'
df['Degree'] = df['Degree'].round().astype(int)
one_hot = pd.get_dummies(df['Degree'])
one_hot.columns = ['Degree_Bachelor', 'Degree_Master', 'Degree_Doctorate', 'Degree_High_School']
for label in one_hot.columns:
    one_hot[label] = one_hot[label].astype(int)
df = df.drop('Degree', axis=1)
df = pd.concat([df, one_hot], axis=1)
print(df.head())


df.to_csv('../data/cleaned_student_depression_dataset.csv', index=False) # save cleaned dataset