import pandas as pd

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
df.drop(columns=['Profession'], inplace=True)
df.drop(columns=['Work Pressure'], inplace=True) # This column is filled with 0.0s
df.drop(columns=['Job Satisfaction'], inplace=True)
df.drop(columns=['Degree'], inplace=True) # TODO: maybe chanbge to one hot encoding?
 
# Look at unique values in each column
for column in df.columns:
    print(f"Unique values in '{column}':")
    print(df[column].unique())
    print('-' * 40)

# Check for missing values, the dataset uses the value 'Others' to indicate missing values, except for 'Finintial Stress' column which uses '?'

df = df[~df.apply(lambda row: row.astype(str).eq('Others').any(), axis=1)] # remove rows with 'Others'
df = df[~df.apply(lambda row: row.astype(str).eq('?').any(), axis=1)]      # remove rows with '?'


rows_with_bad_labels = df[df.apply(lambda col: col.astype(str).isin(['Others', '?'])).any(axis=1)]
print(rows_with_bad_labels.shape[0]==0) # check if there are still any rows with bad labels


# Convert categorical columns to numerical values (by one hot encoding)
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Family History of Mental Illness'] = df['Family History of Mental Illness'].map({'No': 0, 'Yes': 1})
df['Have you ever had suicidal thoughts ?'] = df['Have you ever had suicidal thoughts ?'].map({'No': 0, 'Yes': 1})

# TODO: check if this encoding is valuable/correct
df['Sleep Duration'] = df['Sleep Duration'].map({'\'Less than 5 hours\'': 0, 
                                                 '\'5-6 hours\'': 1, 
                                                 '\'7-8 hours\'': 2, 
                                                 '\'More than 8 hours\'': 3})

df['Dietary Habits'] = df['Dietary Habits'].map({'Unhealthy': -1,
                                                 'Moderate': 0, 
                                                 'Healthy': 1})

print(df.head())
print(df.tail())

df.to_csv('../data/cleaned_student_depression_dataset.csv', index=False) # save cleaned dataset