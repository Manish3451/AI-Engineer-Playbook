## Linear regression using normal equation
import numpy as np 
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class LinearRegression:
    """Linear Regression using Normal Equation: θ = (X^T X)^(-1) X^T y"""

    def __init__(self):
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        n_samples = X.shape[0]
        X_bias = np.c_[np.ones(n_samples), X]

        theta = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y

        self.bias = theta[0]
        self.weights = theta[1:]

    def predict(self, X):
        return X @ self.weights + self.bias


if __name__ == "__main__":
    X, y = make_regression(n_samples=100, n_features=3, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"mse: {mean_squared_error(y_test, y_pred)}")
    print(f"R²: {r2_score(y_test, y_pred):.3f}")
    print(f"Weights: {model.weights[:2]}...")