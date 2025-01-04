import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np

digits = datasets.load_digits()

X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

# Initialiser le classifieur SVC (One-vs-One)
classifier = SVC(gamma=0.001, C=100.)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)

def plot_random_predictions():
    indices = np.random.choice(range(len(X_test)), size=20, replace=False)
    fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    axes = axes.flatten()
    for ax, idx in zip(axes, indices):
        ax.set_axis_off()
        ax.imshow(X_test[idx].reshape(8, 8), cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Pred: {y_pred[idx]}")
    plt.tight_layout()
    plt.show()

plot_random_predictions()
