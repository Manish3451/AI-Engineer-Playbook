import numpy as np 
import pandas as pd 


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
