##batch gradient descent for linear regression
import numpy as np 
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



class BatchGradientDescent:
    """Uses ALL samples in each iteration."""
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iterations):
            y_pred = X @ self.weights + self.bias

            dw = (1/n_samples) * X.T @ (y_pred - y)
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self
    


    def predict(self, X):
        return X @ self.weights + self.bias
    

if __name__ == "__main__":
        X, y = make_regression(n_samples=100, n_features=3, noise=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Standardize
        X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
        X_test = (X_test - X_train.mean(axis=0)) / X_train.std(axis=0)
        
        model = BatchGradientDescent(learning_rate=0.1, n_iterations=1000)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        print(f"Batch GD - MSE: {mean_squared_error(y_test, y_pred):.2f}, R²: {r2_score(y_test, y_pred):.3f}")