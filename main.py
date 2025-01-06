from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=10000)

log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
print(classification_report(y_test, y_pred))

#afficher quelques prédictions correctes et incorrects

correct_indices = np.where(y_pred == y_test)[0]
incorrect_indices = np.where(y_pred != y_test)[0]

np.random.shuffle(correct_indices)
np.random.shuffle(incorrect_indices)

random_correct_indices = correct_indices[:5]
random_incorrect_indices = incorrect_indices[:5]

plt.figure(figsize=(10, 5))
for i, index in enumerate(random_correct_indices):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[index].reshape(8, 8), cmap=plt.cm.gray)
    plt.title(f'Pred: {y_pred[index]} | True: {y_test[index]}', color='green')
    plt.axis('off')

for i, index in enumerate(random_incorrect_indices):
    plt.subplot(2, 5, i + 6)
    plt.imshow(X_test[index].reshape(8, 8), cmap=plt.cm.gray)
    plt.title(f'Pred: {y_pred[index]} | True: {y_test[index]}', color='red')
    plt.axis('off')

plt.tight_layout()
plt.show()