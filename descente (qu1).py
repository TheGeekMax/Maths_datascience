import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from descente_stochastique import GradientDescent

def sigmoid(z):
    z = np.clip(z, -500, 500)  # Limite les valeurs pour éviter le débordement
    return 1 / (1 + np.exp(-z))

def compute_gradient(weights, x_batch, y_batch):
    m = x_batch.shape[0]
    predictions = sigmoid(np.dot(x_batch, weights))
    errors = predictions - y_batch
    gradient = np.dot(x_batch.T, errors) / m
    return gradient

digits = load_digits()
X, y = digits.data, digits.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = np.hstack((np.ones((X.shape[0], 1)), X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_classes = len(np.unique(y))
initial_weights = np.zeros((X_train.shape[1], num_classes))  # Poids pour chaque classe
learning_rate = 0.1
max_iterations = 500

def wrapped_gradient(weights, batch):
    x_batch = batch[:, :-1]  # Toutes les colonnes sauf la dernière
    y_batch = batch[:, -1].astype(int)  # La dernière colonne
    predictions = sigmoid(np.dot(x_batch, weights))  # Prédictions pour chaque classe
    one_hot = np.eye(num_classes)[y_batch]  # Convertir les labels en one-hot encoding
    errors = predictions - one_hot
    gradient = np.dot(x_batch.T, errors) / x_batch.shape[0]
    return gradient

data = np.hstack((X_train, y_train.reshape(-1, 1)))
gd = GradientDescent(gradient=wrapped_gradient, learning_rate=learning_rate, max_iterations=max_iterations, batch_size=32)

weights = gd.descent(initial_weights, data)

y_pred = np.argmax(sigmoid(np.dot(X_test, weights)), axis=1)

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
    # Retirer la colonne de biais avant reshape
    plt.imshow(X_test[index, 1:].reshape(8, 8), cmap=plt.cm.gray)
    plt.title(f'Pred: {y_pred[index]} | True: {y_test[index]}', color='green')
    plt.axis('off')

for i, index in enumerate(random_incorrect_indices):
    plt.subplot(2, 5, i + 6)
    # Retirer la colonne de biais avant reshape
    plt.imshow(X_test[index, 1:].reshape(8, 8), cmap=plt.cm.gray)
    plt.title(f'Pred: {y_pred[index]} | True: {y_test[index]}', color='red')
    plt.axis('off')

plt.tight_layout()
plt.show()
