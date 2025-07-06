"""
melbourne_random_forest.py

A RandomForest Regression example using the Melbourne real estate dataset.

This script demonstrates how to train and evaluate a random forest regressor
to predict house prices based on various features from the dataset.

Author: Amit Elia
Date: 2023-10-05
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset
data = pd.read_csv('melb_data.csv')
# Display the first few rows of the dataset
melbourne_data.describe()
# Select features and target variable
y = melbourne_data.Price
features = ['MSSubClass',
'LotArea',
'OverallQual',
'OverallCond',
'YearBuilt',
'YearRemodAdd',
'1stFlrSF',
'2ndFlrSF',
'LowQualFinSF',
'GrLivArea',
'FullBath',
'HalfBath',
'BedroomAbvGr',
'KitchenAbvGr',
'TotRmsAbvGrd',
'Fireplaces',
'WoodDeckSF',
'OpenPorchSF',
'EnclosedPorch',
'3SsnPorch',
'ScreenPorch',
'PoolArea',
'MiscVal',
'MoSold',
'YrSold']

X = melbourne_data[features]
X.describe()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
# Create and train the Random Forest Regressor
melbourne_model = RandomForestRegressor(random_state=1)
melbourne_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = melbourne_model.predict(X_test)
# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
print(f'R^2 Score: {r2}')

# Display the feature importances
feature_importances = pd.Series(melbourne_model.feature_importances_, index=X.columns)
print("Feature Importances:")
print(feature_importances.sort_values(ascending=False))





def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

