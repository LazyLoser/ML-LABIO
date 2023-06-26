import pandas as pd

# Load the automobile dataset into a DataFrame
df = pd.read_csv('Automobile_data.csv')

# Identify duplicate rows
duplicates = df[df.duplicated()]

# View the duplicate rows
print("Duplicate Rows:")
print(duplicates)

# Delete duplicate rows
df.drop_duplicates(inplace=True)

# Verify the deletion
print("Cleaned DataFrame:")
print(df)

import pandas as pd

# Load the automobile dataset into a DataFrame
df = pd.read_csv('Automobile_data.csv')

# Identify columns with a single value
single_value_columns = []
for column in df.columns:
    if df[column].nunique() == 1:
        single_value_columns.append(column)

# Delete the identified columns
df.drop(single_value_columns, axis=1, inplace=True)

# Verify the deletion
print("Updated DataFrame:")
print(df)
