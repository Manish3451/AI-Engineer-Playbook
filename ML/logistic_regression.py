import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias

            y_pred = self.sigmoid(linear_model)

            dw = (1/n_samples) * np.dot(X.T,(y_pred-y))

            db = (1/n_samples) * np.sum(y_pred-y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = -np.mean(y*np.log(y_pred)+(1-y)*np.log(1-y_pred))
            self.losses.append(loss)


    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X) 
        return (probabilities >= threshold).astype(int)
    

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Train the model
model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Calculate accuracy
accuracy = np.mean(y == y_pred)
print(f"Accuracy: {accuracy:.2f}")




# Visualizations
plt.figure(figsize=(15, 5))

# Plot 1: Training loss
plt.subplot(1, 3, 1)
plt.plot(model.losses)
plt.title('Training Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid(True)

# Plot 2: Decision boundary
plt.subplot(1, 3, 2)
# Create mesh for decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
plt.title('Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot 3: Predictions vs Actual
plt.subplot(1, 3, 3)
plt.scatter(range(len(y)), y, color='blue', label='Actual', alpha=0.6)
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted', alpha=0.6)
plt.title('Predictions vs Actual')
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.legend()

plt.tight_layout()
plt.show()

# Print model parameters
print(f"\nModel weights: {model.weights}")
print(f"Model bias: {model.bias:.4f}")