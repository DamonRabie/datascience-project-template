import numpy as np

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import precision_score, recall_score, precision_recall_curve, roc_curve, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, cross_val_predict


def get_train_test(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def compute_precision_recall(labels, predictions, p=0.5):
    return precision_score(labels, predictions > p), recall_score(labels, predictions > p)

def compute_f1_score(labels, predictions, p=0.5):
    return f1_score(labels, predictions > p)

def find_opt_threshold(labels, predictions):
    best_threshold = 0
    best_f1_score = 0

    # Find the optimum threshold with the highest F1 score
    for threshold in np.arange(0.1, 1.0, 0.1):
        predictions_thresholded = (predictions >= threshold).astype(int)
        f1 = f1_score(labels, predictions_thresholded)

        if f1 > best_f1_score:
            best_f1_score = f1
            best_threshold = threshold

    return best_threshold

def compute_cross_val_predict_scores(model, X, y, cv=3):
    if hasattr(model, "decision_function"):
        return cross_val_predict(model, X, y, cv=cv, method="decision_function")
    elif hasattr(model, "predict_proba"):
        return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    else:
        raise "Model does not have either decision_function or predict_proba attributes"

def plot_metrics(history, **kwargs):
    metrics = ['loss', 'prc', 'precision', 'recall']
    fig = plt.figure()
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric], label='Train', **kwargs)
        plt.plot(history.epoch, history.history['val_'+metric], linestyle="--", label='Val', **kwargs)
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        else:
            plt.ylim([0,1.05])

        plt.legend()

def plot_confusion_matrix(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    fig = plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

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


def plot_prc(name, labels, predictions=None, features=None, model=None, cv=3, **kwargs):
    if model is not None:
        predictions = compute_cross_val_predict_scores(model, features, labels, cv)

    precisions, recalls, thresholds = precision_recall_curve(labels, predictions)

    plt.plot(recalls, precisions, label=name, linewidth=2, **kwargs)
    plt.legend(loc="lower right")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(color='grey', linestyle='--', linewidth=0.5, which="both")

def plot_roc(name, labels, predictions=None, features=None, model=None, cv=3, random_curve=True, **kwargs):
    if model is not None:
        predictions = compute_cross_val_predict_scores(model, features, labels, cv)

    fpr, tpr, thresholds = roc_curve(labels, predictions)

    plt.plot(fpr, tpr, label=name, linewidth=2, **kwargs)
    if random_curve:
        plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")

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
