import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initialize logistic regression model
        
        Parameters:
        learning_rate (float): Learning rate for gradient descent
        n_iterations (int): Number of iterations for training
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def sigmoid(self, z):
        """
        Sigmoid activation function
        
        Parameters:
        z: Linear combination of inputs and weights
        
        Returns:
        Probability between 0 and 1
        """
        # Clip z to prevent overflow in exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def initialize_parameters(self, n_features):
        """Initialize weights and bias with zeros"""
        self.weights = np.zeros(n_features)
        self.bias = 0
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute binary cross-entropy loss
        
        Parameters:
        y_true: True labels
        y_pred: Predicted probabilities
        
        Returns:
        Loss value
        """
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Binary cross-entropy loss
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def fit(self, X, y, verbose=False):
        """
        Train the logistic regression model using gradient descent
        
        Parameters:
        X: Feature matrix (n_samples, n_features)
        y: Target labels (n_samples,)
        verbose: Whether to print progress
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.initialize_parameters(n_features)
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            
            # Compute loss
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)
            
            # Backward pass (compute gradients)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if verbose and i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss:.4f}")
    
    def predict_proba(self, X):
        """
        Predict probabilities
        
        Parameters:
        X: Feature matrix
        
        Returns:
        Probability of positive class
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels
        
        Parameters:
        X: Feature matrix
        threshold: Decision threshold
        
        Returns:
        Binary predictions
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def accuracy(self, y_true, y_pred):
        """Compute accuracy score"""
        return np.mean(y_true == y_pred)

# Example usage and demonstration
def demonstrate_logistic_regression():
    # Generate sample data
    np.random.seed(42)
    
    # Create two classes
    n_samples = 1000
    n_features = 2
    
    # Class 0: centered at (0, 0)
    X_class0 = np.random.randn(n_samples // 2, n_features) + np.array([0, 0])
    y_class0 = np.zeros(n_samples // 2)
    
    # Class 1: centered at (2, 2)
    X_class1 = np.random.randn(n_samples // 2, n_features) + np.array([2, 2])
    y_class1 = np.ones(n_samples // 2)
    
    # Combine data
    X = np.vstack([X_class0, X_class1])
    y = np.hstack([y_class0, y_class1])
    
    # Add bias term and shuffle data
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    # Split into train and test
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create and train model
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X_train, y_train, verbose=True)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Evaluate model
    accuracy = model.accuracy(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Training loss
    plt.subplot(1, 3, 1)
    plt.plot(model.loss_history)
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot 2: Decision boundary
    plt.subplot(1, 3, 2)
    
    # Create mesh for decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Predict probabilities for mesh points
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and data points
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.RdBu, edgecolors='black')
    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Plot 3: Prediction probabilities
    plt.subplot(1, 3, 3)
    plt.scatter(range(len(y_pred_proba)), y_pred_proba, c=y_test, cmap=plt.cm.coolwarm)
    plt.axhline(y=0.5, color='red', linestyle='--', label='Decision Threshold')
    plt.title('Prediction Probabilities')
    plt.xlabel('Sample Index')
    plt.ylabel('Probability')
    plt.legend()
    plt.colorbar(label='True Label')
    
    plt.tight_layout()
    plt.show()
    
    # Print model parameters
    print(f"\nModel Parameters:")
    print(f"Weights: {model.weights}")
    print(f"Bias: {model.bias:.4f}")

# Additional utility: Softmax for multi-class classification
def softmax(z):
    """Softmax function for multi-class classification"""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

if __name__ == "__main__":
    demonstrate_logistic_regression()