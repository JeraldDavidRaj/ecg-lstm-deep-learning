import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(model, X_test, y_test):
    os.makedirs("../results/metrics", exist_ok=True)
    os.makedirs("../results/plots", exist_ok=True)

    # Evaluate model
    loss, acc = model.evaluate(X_test, y_test)
    with open("../results/metrics/accuracy.txt", "w") as f:
        f.write(f"Test Accuracy: {acc * 100:.2f}%")

    # Predict and classification report
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    report = classification_report(y_test, y_pred_classes)
    with open("../results/metrics/classification_report.txt", "w") as f:
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig("../results/plots/confusion_matrix.png")
    plt.close()
