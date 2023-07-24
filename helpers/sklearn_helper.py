import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import precision_score, recall_score, precision_recall_curve, roc_curve, confusion_matrix, \
    f1_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_predict


# Function to split data into training and testing sets
def get_train_test(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# Function to compute precision and recall scores for a given threshold
def compute_precision_recall(labels, predictions, p=0.5):
    return precision_score(labels, predictions > p), recall_score(labels, predictions > p)


# Function to compute F1 score for a given threshold
def compute_f1_score(labels, predictions, p=0.5):
    return f1_score(labels, predictions > p)


# Function to print classification report for a given threshold
def print_classification_report(labels, predictions, p=0.5, names=['class_0', 'class_1']):
    print(classification_report(y_true=labels, y_pred=predictions >= p, target_names=names))


# Function to find the optimal threshold that maximizes the F1 score
def find_opt_threshold(labels, predictions, metric='f1'):
    best_threshold = 0
    best_score = 0

    # Find the optimum threshold with the highest specified metric score
    for threshold in np.arange(0.1, 1.0, 0.05):
        predictions_thresholded = (predictions >= threshold).astype(int)
        if metric == 'f1':
            m = f1_score(labels, predictions_thresholded)
        elif metric == 'recall':
            m = recall_score(labels, predictions_thresholded)
        elif metric == 'precision':
            m = precision_score(labels, predictions_thresholded)
        else:
            raise ValueError("Wrong metric")  # Corrected the exception type to ValueError

        if m > best_score:
            best_score = m
            best_threshold = threshold

    return best_threshold


# Function to compute cross-validated predictions scores
def compute_cross_val_predict_scores(model, X, y, cv=3):
    if hasattr(model, "decision_function"):
        return cross_val_predict(model, X, y, cv=cv, method="decision_function")
    elif hasattr(model, "predict_proba"):
        return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    else:
        raise ValueError(
            "Model does not have either decision_function or predict_proba attributes")  # Corrected the exception type to ValueError


def plot_metrics(history, metrics=None, nrows=2, ncols=2, figsize=(12, 10), **kwargs):
    """
    Plot training metrics over epochs from model training history.

    Parameters:
        history (keras.callbacks.History): Model training history containing the metrics.
        metrics (list): List of metrics to plot. If None, ['loss', 'prc', 'precision', 'recall'] will be used.
        nrows (int): Number of rows in the subplot grid.
        ncols (int): Number of columns in the subplot grid.
        figsize (tuple): Figure size (width, height) in inches.
        **kwargs: Additional keyword arguments to customize the plot.

    Returns:
        matplotlib.figure.Figure: The created matplotlib figure.
    """
    if metrics is None:
        metrics = ['loss', 'prc', 'precision', 'recall']

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows, ncols)

    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        ax = fig.add_subplot(gs[n // ncols, n % ncols])
        ax.plot(history.epoch, history.history[metric], label='Train', **kwargs)
        ax.plot(history.epoch, history.history['val_' + metric], linestyle="--", label='Val', **kwargs)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(name)

        if metric == 'loss':
            ax.set_ylim([0, ax.get_ylim()[1]])
        else:
            ax.set_ylim([0, 1.05])

        ax.legend()

    plt.tight_layout()

    return fig


# Function to plot the confusion matrix
def plot_confusion_matrix(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    fig = plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')


# Function to plot precision-recall curves with different thresholds
def plot_prc_threshold(name, labels, predictions=None, features=None, model=None, cv=3, **kwargs):
    if model is not None:
        predictions = compute_cross_val_predict_scores(model, features, labels, cv)

    precisions, recalls, thresholds = precision_recall_curve(labels, predictions)

    plt.figure()
    plt.plot(thresholds, precisions[:-1], label=f'precision_{name}', linewidth=2, **kwargs)
    plt.plot(thresholds, recalls[:-1], label=f'recall_{name}', linewidth=2, linestyle='--', **kwargs)
    plt.legend(loc="lower right")
    plt.xlabel("Threshold")
    plt.ylabel("Precision/Recall Score")
    plt.grid(color='grey', linestyle='--', linewidth=0.5, which="both")


# Function to plot precision-recall curve
def plot_prc(name, labels, predictions=None, features=None, model=None, cv=3, **kwargs):
    if model is not None:
        predictions = compute_cross_val_predict_scores(model, features, labels, cv)

    precisions, recalls, thresholds = precision_recall_curve(labels, predictions)

    plt.plot(recalls, precisions, label=name, linewidth=2, **kwargs)
    plt.legend(loc="lower right")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(color='grey', linestyle='--', linewidth=0.5, which="both")


# Function to plot ROC curve
def plot_roc(name, labels, predictions=None, features=None, model=None, cv=3, random_curve=True, **kwargs):
    if model is not None:
        predictions = compute_cross_val_predict_scores(model, features, labels, cv)

    fpr, tpr, thresholds = roc_curve(labels, predictions)

    plt.plot(fpr, tpr, label=name, linewidth=2, **kwargs)
    if random_curve:
        plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")

    # Adding an arrow and text to indicate the direction of higher threshold
    plt.gca().add_patch(patches.FancyArrowPatch(
        (0.20, 0.89), (0.07, 0.70),
        connectionstyle="arc3,rad=.4",
        arrowstyle="Simple, tail_width=1.5, head_width=8, head_length=10",
        color="#444444"))
    plt.text(0.12, 0.71, "Higher\nthreshold", color="#333333")

    plt.xlabel('False Positive Rate (Fall-Out)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.grid()
    plt.axis([-0.01, 1.01, -0.01, 1.01])
    plt.legend(loc="lower right", fontsize=13)
