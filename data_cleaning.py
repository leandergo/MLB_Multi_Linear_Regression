import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from random import sample


# Load the data
batting = pd.read_csv('/Users/Leander/Desktop/Projects/mlb_multi_linear_regression/lahman_1871-2023_csv/lahman_1871-2023_csv/Batting.csv')
batting = batting.drop(["G_batting", "G_old"], axis=1)
salaries = pd.read_csv('/Users/Leander/Desktop/Projects/mlb_multi_linear_regression/lahman_1871-2023_csv/lahman_1871-2023_csv/Salaries.csv')

salary_combo = pd.merge(batting, salaries, on=['playerID', 'yearID', 'teamID', 'lgID'], how='inner')

df = salary_combo.groupby(["playerID", "yearID", "salary"]).sum().reset_index()
df = df.drop(["teamID", "lgID"], axis=1)
df = df.sort_values(["playerID", "yearID"])
player_counts = df['playerID'].value_counts()
df = df[df['playerID'].isin(player_counts[player_counts >= 3].index)]
df = df[df['salary'] > 0]

AT_BAT_MINIMUM = 75

df = df[df['AB'] >= AT_BAT_MINIMUM].reset_index(drop=True)
for idx in df.index:
    salary = df.at[idx, "salary"]
    if idx == df.index[-1]:
        df.at[idx, "salary_bump"] = np.nan
        break
    if df.at[idx, "playerID"] == df.at[idx+1, "playerID"]:
        df.at[idx, "salary_bump"] = (df.at[idx+1, "salary"] - df.at[idx, "salary"]) / df.at[idx, "salary"]
    else:
        df.at[idx, "salary_bump"] = np.nan
df = df.dropna().reset_index(drop=True)
df = df.sort_values(["yearID"])

scaler = StandardScaler()

# Select the columns to standardize
columns_to_standardize = ['salary', 'stint', 'G', 'AB', 'R', 'H', '2B',
       '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF',
       'GIDP']

# Standardize and replace original columns
df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])

year = 2014

# Split the DataFrame into two parts based on the threshold
train = df[df['yearID'] < year].reset_index(drop=True)
test = df[df['yearID'] >= year].reset_index(drop=True)
### Some EDA to determine features I want to use
# salary_bump = train['salary_bump']
# for column in  ['yearID']:
#     random_sample = sample(range(len(train)), 1000)
#     plt.scatter(train.iloc[random_sample][column], salary_bump[random_sample])
#     plt.xlabel(column)
#     plt.ylabel('Salary Bump')
#     plt.show()

train.to_csv('/Users/Leander/Desktop/Projects/mlb_multi_linear_regressiong/train.csv', index=False)
test.to_csv('/Users/Leander/Desktop/Projects/mlb_multi_linear_regression/test.csv', index=False)