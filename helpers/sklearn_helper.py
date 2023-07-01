import matplotlib.patches as patches
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split, cross_val_predict


def get_train_test(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def print_precision_recall(y, y_pred, text=''):
    print(f"{text} Precision: {precision_score(y, y_pred):.4%}")
    print(f"{text} Recall: {recall_score(y, y_pred):.4%}")


def compute_cross_val_predict_scores(model, X, y, cv=3):
    if hasattr(model, "decision_function"):
        return cross_val_predict(model, X, y, cv=cv, method="decision_function")
    elif hasattr(model, "predict_proba"):
        return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    else:
        raise "Model does not have either decision_function or predict_proba attributes"


def plot_precision_recall_curve(model, X, y, cv=3):
    y_scores = compute_cross_val_predict_scores(model, X, y, cv)
    precisions, recalls, thresholds = precision_recall_curve(y, y_scores)

    plt.figure(1)
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="lower right")
    plt.xlabel("Precision/Recall Score")
    plt.ylabel("Threshold")
    plt.grid(color='grey', linestyle='--', linewidth=0.5, which="both")

    plt.figure(2)
    plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")
    plt.legend(loc="lower right")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(color='grey', linestyle='--', linewidth=0.5, which="both")

    plt.show()


def plot_roc_curve(model, X, y, cv=3):
    y_scores = compute_cross_val_predict_scores(model, X, y, cv)

    fpr, tpr, thresholds = roc_curve(y, y_scores)

    plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
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
    plt.axis([0, 1, 0, 1])
    plt.legend(loc="lower right", fontsize=13)

    plt.show()
