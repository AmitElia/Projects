"""
XGBoost Classification Approach: Introvert vs Extrovert Prediction

Objective:
	Predict whether a person is an Introvert or Extrovert based on their social behavior and personality traits using the XGBoost machine learning algorithm.

Author: Amit Elia
Date: 2023-10-05
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, plot_importance
import lightgbm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import optuna
from sklearn.pipeline import Pipeline
import xgboost as xgb
optuna.logging.set_verbosity(optuna.logging.WARNING)
import matplotlib.pyplot as plt

train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

train.describe()
train.info()


train = train.drop(columns=['id'])
X_test = test.drop(columns=['id'])

cols_with_missing = [col for col in train.columns
                     if train[col].isnull().any()]

# Change NA numerical values with mean
numerical = train.select_dtypes(include='number').columns
for cat in numerical:
    train[cat] = train[cat].fillna(train[cat].mean())
    X_test[cat] = X_test[cat].fillna(X_test[cat].mean())

# Change NA categorial values to 'missing'
categorical = ['Stage_fear', 'Drained_after_socializing', 'Personality']
for cat in categorical:
    train[cat] = train[cat].fillna('Missing')
    if cat != 'Personality':
        X_test[cat] = X_test[cat].fillna('Missing')

# Convert categorial (object) values to numerical values using label encoder
le_dict = {}
for cat in categorical:
    le = LabelEncoder()
    train[cat] = le.fit_transform(train[cat].astype(str))
    if cat in X_test.columns:
        X_test[cat] = le.transform(X_test[cat].astype(str))
    le_dict[cat] = le

train.head(10)

# Split the dataset into features and target variable
X = train.drop(columns=['Personality'])
y = train['Personality']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=83, test_size=0.2, stratify=y)


#create optuna study for XGboost
def objective(trial):
	params = {
		'objective': 'binary:logistic',
		'eval_metric': 'auc',
		'tree_method': 'gpu_hist',
		'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
		'max_depth': trial.suggest_int('max_depth', 3, 10),
		'subsample': trial.suggest_float('subsample', 0.5, 1.0),
		'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
		'lambda': trial.suggest_float('lambda', 1e-8, 10.0),
		'alpha': trial.suggest_float('alpha', 1e-8, 10.0),
        'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42,
		'n_estimators': 10000,
	}
    # If the booster is 'dart', add additional parameters
	if params['booster'] == 'dart':
		params['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
		params['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
		params['rate_drop'] = trial.suggest_float('rate_drop', 0.0, 0.5)
		params['skip_drop'] = trial.suggest_float('skip_drop', 0.0, 0.5)

	# Create and train the XGBoost model
	model = XGBClassifier(**params)
	model.fit(
        X_train, y_train,
    	eval_set=[(X_test, y_test)],
		early_stopping_rounds=50,
		verbose=False
	)
	# Make predictions and calculate AUC
	preds = model.predict_proba(X_test)[:, 1]
	auc = roc_auc_score(y_test, preds)
	return auc

# Create an Optuna study and optimize the objective function
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective, n_trials=100)
best_params_xgb = study_xgb.best_params
print("Best parameters for XGBoost:", best_params_xgb)
best_value_xgb = study_xgb.best_value
print("Best AUC value for XGBoost:", best_value_xgb)

# Train the best XGBoost model with the best parameters
best_xgb = best_xgb.XGBClassifier(
    **best_params_xgb,
    random_state=42,
    use_label_encoder=False,
    eval_metric='auc'
)

# Fit the model on the training data
best_xgb.fit(
    X_train, 
    y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=50,
    verbose=False
)

# Make predictions on the test set
y_pred = best_xgb.predict(X_test)
y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, y_pred_proba)
# Print evaluation metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
values = [accuracy, precision, recall, f1, roc_auc]
metrics_df = pd.DataFrame({'Metric': metrics, 'Value': values})
print(metrics_df)


#plot accuracy
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Metric', y='Value', data=metrics_df, palette='viridis')
plt.title('Model Evaluation Metrics')
plt.xlabel('Metrics')
plt.ylabel('Value')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for p in ax.patches:
	ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
				ha='center', va='bottom', fontsize=12, color='black', xytext=(0, 5),
				textcoords='offset points')


#prepare for submission
pred_probs = best_model_xgb.predict_proba(X_test)

# Get the class with the highest probability
final_preds = np.argmax(pred_probs, axis=1)

# Use the fitted LabelEncoder to inverse transform numerical predictions back to original labels
final_labels = le.inverse_transform(final_preds)

# Create the submission DataFrame
submission_df = pd.DataFrame({
    'id': test['id'],
    'Personality': final_labels
})

# Save the submission DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)