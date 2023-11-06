import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix, \
    precision_recall_fscore_support, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.utils.multiclass import unique_labels


def get_train_test(X, y, test_size=0.2, random_state=42, stratify=True):
    """
    Split the data into training and testing sets while preserving the class distribution (stratified sampling).

    Parameters:
        X (array-like): The input features.
        y (array-like): The target labels.
        test_size (float or int): The proportion of the dataset to include in the test split,
                                  or the number of samples to include in the test split.
        random_state (int or RandomState instance): Controls the randomness of the data shuffling.
        stratify (bool): Whether to use stratified sampling based on the target labels or not.

    Returns:
        X_train (array-like): The training features.
        X_test (array-like): The testing features.
        y_train (array-like): The training labels.
        y_test (array-like): The testing labels.
    """
    if stratify:
        # Use stratified sampling to ensure balanced class distribution in training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                            stratify=y)
    else:
        # Use regular random sampling without preserving class distribution
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


def compute_classification_metrics(labels, predictions, threshold=0.5):
    """
    Compute precision, recall, F1-score, and support for binary classification.

    Parameters:
        labels (array-like): True binary labels.
        predictions (array-like): Predicted probabilities or scores.
        threshold (float, optional): Decision threshold for converting scores to binary predictions.

    Returns:
        dict: A dictionary containing precision, recall, F1-score, and support values for both classes.
    """
    binary_predictions = (predictions >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(labels, binary_predictions, average=None)
    metrics_dict = {
        'class_0': {'precision': precision[0], 'recall': recall[0], 'f1-score': f1[0], 'support': support[0]},
        'class_1': {'precision': precision[1], 'recall': recall[1], 'f1-score': f1[1], 'support': support[1]}
    }
    return metrics_dict


def find_opt_threshold(labels, predictions, metric='f1'):
    """
    Find the optimal threshold that maximizes the specified metric.

    Parameters:
        labels (array-like): True labels.
        predictions (array-like): Predicted probabilities or scores from the model.
        metric (str): The metric to optimize. Options: 'f1' (default), 'recall', 'precision'.

    Returns:
        float: The optimal threshold value that maximizes the specified metric.
    """

    # Define a dictionary to map the metric name to the corresponding scoring function
    metric_to_scoring = {
        'f1': f1_score,
        'recall': recall_score,
        'precision': precision_score
    }

    if metric not in metric_to_scoring:
        raise ValueError("Invalid metric specified. Choose from 'f1', 'recall', or 'precision'.")

    best_threshold = 0
    best_score = 0

    # Iterate over a range of thresholds and find the one that maximizes the metric
    for threshold in np.arange(0.1, 1.0, 0.05):
        # Convert probabilities to binary predictions based on the threshold
        predictions_thresholded = (predictions >= threshold).astype(int)

        # Compute the specified metric score
        scoring_function = metric_to_scoring[metric]
        m = scoring_function(labels, predictions_thresholded)

        # Update the best threshold and score if the current threshold performs better
        if m > best_score:
            best_score = m
            best_threshold = threshold

    return best_threshold


def compute_cross_val_predict_scores(model, X, y, cv=3, method='auto'):
    """
    Compute cross-validated prediction scores for a given model.

    Parameters:
        model (object): The machine learning model with either 'decision_function' or 'predict_proba' attributes.
        X (array-like): The input features for the data.
        y (array-like): The target labels for the data.
        cv (int, cross-validation generator, or an iterable): Determines the cross-validation strategy.
        method (str, optional): The method to be used for obtaining prediction scores.
                                'auto' (default) will automatically choose 'decision_function' or 'predict_proba'
                                based on the model's capabilities.
                                'decision_function' and 'predict_proba' force the corresponding method.

    Returns:
        ndarray: The cross-validated prediction scores for the positive class.
    """

    # Check if the given model has either 'decision_function' or 'predict_proba' attributes
    if not hasattr(model, "decision_function") and not hasattr(model, "predict_proba"):
        raise ValueError("Model does not have either 'decision_function' or 'predict_proba' attributes")

    # Determine the method to be used for obtaining prediction scores
    if method == 'auto':
        if hasattr(model, 'decision_function'):
            method = 'decision_function'
        elif hasattr(model, 'predict_proba'):
            method = 'predict_proba'
    elif method not in ('decision_function', 'predict_proba'):
        raise ValueError("Invalid value for 'method', it should be 'auto', 'decision_function', or 'predict_proba'")

    # Compute cross-validated prediction scores based on the selected method
    if method == 'decision_function':
        return cross_val_predict(model, X, y, cv=cv, method='decision_function')
    elif method == 'predict_proba':
        return cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]


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


def plot_confusion_matrix(labels, predictions, p=0.5, normalize=False, cmap='Blues', ax=None):
    """
    Plot the confusion matrix for binary classification.

    Parameters:
        labels (array-like): True labels of the data.
        predictions (array-like): Predicted probabilities or scores of the data.
        p (float, optional): Threshold value to convert probabilities/scores to binary predictions. Default is 0.5.
        normalize (bool, optional): If True, the confusion matrix will be normalized. Default is False.
        cmap (str, optional): Colormap for the heatmap. Default is 'Blues'.
        ax (matplotlib Axes, optional): The Axes object to plot the confusion matrix. If not provided, a new figure will be created.

    Returns:
        matplotlib Axes: The Axes object containing the plotted confusion matrix.
    """
    cm = confusion_matrix(labels, predictions > p)
    classes = unique_labels(labels, predictions > p)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion matrix',
           ylabel='True label',
           xlabel='Predicted label')

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylim(len(classes) - 0.5, -0.5)
    plt.tight_layout()

    return ax


def plot_prc_curve(name, labels, predictions=None, features=None, model=None, cv=3, threshold=True, **kwargs):
    """
    Plot Precision-Recall Curve for a binary classification model.

    Parameters:
        name (str): The name of the curve for the legend.
        labels (array-like): The true labels of the data.
        predictions (array-like, optional): The predicted probabilities or scores from the model.
        features (array-like, optional): The input features if 'model' is provided.
        model (object, optional): The machine learning model. If provided, predictions will be computed using
                                  cross-validation.
        cv (int, optional): Number of cross-validation folds. Default is 3.
        threshold (bool, optional): Whether to plot threshold values on the curve. Default is True.
        **kwargs: Additional keyword arguments to pass to plt.plot().

    Returns:
        None
    """
    if model is not None:
        predictions = compute_cross_val_predict_scores(model, features, labels, cv)

    precisions, recalls, thresholds = precision_recall_curve(labels, predictions)

    # Plot Precision-Recall Curve
    plt.plot(recalls, precisions, label=name, linewidth=2, **kwargs)

    if threshold:
        # Plot threshold values on the curve
        for i, thr in enumerate(thresholds):
            plt.scatter(recalls[i], precisions[i], c='r', marker='x', s=50)
            plt.text(recalls[i], precisions[i], f"{thr:.2f}", fontsize=8, verticalalignment='bottom')

    plt.legend(loc="lower right")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(color='grey', linestyle='--', linewidth=0.5, which="both")


def plot_roc_curve(name, labels, predictions=None, features=None, model=None, cv=3, random_curve=True, **kwargs):
    """
    Plot Receiver Operating Characteristic (ROC) curve for a binary classification model.

    Parameters:
        name (str): The name of the curve for the legend.
        labels (array-like): The true labels of the data.
        predictions (array-like, optional): The predicted probabilities or scores from the model.
        features (array-like, optional): The input features if 'model' is provided.
        model (object, optional): The machine learning model. If provided, predictions will be computed using
                                  cross-validation.
        cv (int, optional): Number of cross-validation folds. Default is 3.
        random_curve (bool, optional): Whether to plot the ROC curve of a random classifier. Default is True.
        **kwargs: Additional keyword arguments to pass to plt.plot().

    Returns:
        None
    """
    if model is not None:
        predictions = compute_cross_val_predict_scores(model, features, labels, cv)

    fpr, tpr, thresholds = roc_curve(labels, predictions)

    # Plot ROC Curve
    plt.plot(fpr, tpr, label=name, linewidth=2, **kwargs)

    if random_curve:
        # Plot the ROC curve of a random classifier
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
