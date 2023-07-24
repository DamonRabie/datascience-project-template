# Python Scripts and Utilities
This repository contains a collection of useful Python scripts and utility modules for various tasks. Below is a list of the available scripts along with brief descriptions:

1. [classification_utils](#classification-utils): This script provides utility functions for evaluating binary classification models and plotting various performance metrics. It includes functions for data splitting, computing classification metrics, finding optimal thresholds, and plotting Precision-Recall and Receiver Operating Characteristic (ROC) curves.
2. [sql_server_connection](#sql-server-connection): This script establishes a connection to a Microsoft SQL Server database and provides functions for executing SQL queries and fetching results.
3. [clickhouse_connection](#clickhouse-connection): This script enables a connection to ClickHouse, an open-source columnar database management system, and offers functionalities to execute queries and retrieve data.


# classification utils
[classification_utils.py](./classification_utils.py) <br><br>
This Python code provides a collection of utility functions for evaluating binary classification models and plotting various performance metrics. It includes functions for splitting data into training and testing sets, computing classification metrics, finding optimal thresholds, and plotting Precision-Recall and Receiver Operating Characteristic (ROC) curves.

### Functions Overview

1. `get_train_test(X, y, test_size=0.2, random_state=42, stratify=True)`:
    - Splits the data into training and testing sets while preserving class distribution (stratified sampling).

2. `compute_classification_metrics(labels, predictions, threshold=0.5)`:
    - Computes precision, recall, F1-score, and support for binary classification.

3. `find_opt_threshold(labels, predictions, metric='f1')`:
    - Finds the optimal threshold that maximizes the specified metric.

4. `compute_cross_val_predict_scores(model, X, y, cv=3, method='auto')`:
    - Computes cross-validated prediction scores for a given model.

5. `plot_metrics(history, metrics=None, nrows=2, ncols=2, figsize=(12, 10), **kwargs)`:
    - Plots training metrics over epochs from model training history.

6. `plot_confusion_matrix(labels, predictions, p=0.5, normalize=False, cmap='Blues', ax=None)`:
    - Plots the confusion matrix for binary classification.

7. `plot_prc_curve(name, labels, predictions=None, features=None, model=None, cv=3, threshold=True, **kwargs)`:
    - Plots the Precision-Recall Curve for a binary classification model.

8. `plot_roc_curve(name, labels, predictions=None, features=None, model=None, cv=3, random_curve=True, **kwargs)`:
    - Plots the Receiver Operating Characteristic (ROC) curve for a binary classification model.

### Requirements

To use these functions, you need the following Python libraries:
- `matplotlib`
- `numpy`
- `sklearn`

### Usage

To use the provided functions, simply import the required functions into your Python script or notebook. For example:

```python
import matplotlib.pyplot as plt
from classification_utils import get_train_test, compute_classification_metrics, find_opt_threshold, \
    compute_cross_val_predict_scores, plot_metrics, plot_confusion_matrix, plot_prc_curve, plot_roc_curve

# Example code to use the functions:
# Load your data and create X (input features) and y (target labels) arrays

# Split data into training and testing sets
X_train, X_test, y_train, y_test = get_train_test(X, y, test_size=0.2, random_state=42)

# Train your binary classification model (e.g., using a classifier from scikit-learn)

# Get predictions from the trained model on the test set
predictions = your_model.predict_proba(X_test)[:, 1]

# Compute and print classification metrics
metrics_dict = compute_classification_metrics(y_test, predictions, threshold=0.5)
print(metrics_dict)

# Find the optimal threshold that maximizes the F1-score
optimal_threshold = find_opt_threshold(y_test, predictions, metric='f1')
print("Optimal Threshold:", optimal_threshold)

# Plot training metrics over epochs
history = your_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
plot_metrics(history)

# Plot the confusion matrix
plot_confusion_matrix(y_test, predictions, p=0.5)

# Plot the Precision-Recall Curve
plot_prc_curve("Model", y_test, predictions)

# Plot the ROC curve
plot_roc_curve("Model", y_test, predictions)
```

# SQL Server Connection
[sql_server_connection.py](./sql_server_connection.py) <br><br>
*TO DO*

# ClickHouse Connection
[clickhouse_connection.py](./clickhouse_connection.py) <br><br>
*TO DO*