import pandas as pd
import numpy as np

### Taken from UCSD DSC 40A Homework 4 ###
### http://datahub.ucsd.edu/user-redirect/git-sync?repo=https://github.com/dsc-courses/dsc40a-2024-fa&subPath=homeworks/hw04/hw04-code.ipynb ###
def solve_normal_equations(X, y):
    '''Returns the optimal parameter vector, w*, given a design matrix X and observation vector y.'''
    return np.linalg.solve(X.T @ X, X.T @ y)

def create_design_matrix(df, columns, intercept=True):
    '''Creates a design matrix by taking the specified columns from the DataFrame df.
       Adds a column of all 1s as the first column if intercept is True, which is the default.
       The argument columns should be a list.
    '''
    df = df.copy()
    df['1'] = 1
    if intercept:
        return df[['1'] + columns].values
    else:
        return df[columns].values
    
def mean_squared_error(X, y, w):
    '''Returns the mean squared error of the predictions Xw and observations y.'''
    return np.mean((y - X @ w) ** 2)

### Loading in train and testing data ###
train = pd.read_csv('/Users/Leander/Desktop/Projects/MLB_Multi_Linear_Regression/train.csv')
test = pd.read_csv('/Users/Leander/Desktop/Projects/MLB_Multi_Linear_Regression/test.csv')

### Adding the column for 1/salary becuase in my EDA I saw a decent 1/x relationship ###
train["1 / salary"] = 1 / train["salary"]
test["1 / salary"] = 1 / test["salary"]

### Features I am using ###
feature_columns = ['yearID', 'salary', 'stint', 'G', 'AB', 'R', 'H', '2B',
       '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF',
       'GIDP', '1 / salary']

### Training the model ###
X_train = create_design_matrix(train, feature_columns)
y_train = train['salary_bump']
w_star = solve_normal_equations(X_train, y_train)

### Testing the model ###
X_test = create_design_matrix(test, feature_columns)
y_test = test['salary_bump']

print(f"Training loss: {mean_squared_error(X_train, y_train, w_star)}")
print(f"Test loss: {mean_squared_error(X_test, y_test, w_star)}")