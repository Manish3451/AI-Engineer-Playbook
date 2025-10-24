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
            # Predict with ALL samples
            y_pred = X @ self.weights + self.bias
            
            # Calculate gradients using ALL samples
            dw = (1/n_samples) * X.T @ (y_pred - y)
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        
        return self
    
    def predict(self, X):
        return X @ self.weights + self.bias


# Test
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



##Gradient Descent using SGD
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class StochasticGradientDescent:
    """Uses ONE random sample in each iteration."""
    
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
            # Pick ONE random sample
            idx = np.random.randint(0, n_samples)
            xi = X[idx:idx+1]  # Keep 2D shape
            yi = y[idx:idx+1]
            
            # Predict with ONE sample
            y_pred = xi @ self.weights + self.bias
            
            # Calculate gradients using ONE sample
            dw = xi.T @ (y_pred - yi)
            db = np.sum(y_pred - yi)
            
            # Update parameters
            self.weights -= self.lr * dw.flatten()
            self.bias -= self.lr * db
        
        return self
    
    def predict(self, X):
        return X @ self.weights + self.bias


# Test
if __name__ == "__main__":
    X, y = make_regression(n_samples=100, n_features=3, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Standardize
    X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
    X_test = (X_test - X_train.mean(axis=0)) / X_train.std(axis=0)
    
    model = StochasticGradientDescent(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"SGD - MSE: {mean_squared_error(y_test, y_pred):.2f}, R²: {r2_score(y_test, y_pred):.3f}")




## Mini Batch Gradient Descent 
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class MiniBatchGradientDescent:
    """Uses a BATCH of samples in each iteration."""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, batch_size=32):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iterations):
            # Pick random batch of samples
            indices = np.random.choice(n_samples, self.batch_size, replace=False)
            X_batch = X[indices]
            y_batch = y[indices]
            
            # Predict with batch
            y_pred = X_batch @ self.weights + self.bias
            
            # Calculate gradients using batch
            dw = (1/self.batch_size) * X_batch.T @ (y_pred - y_batch)
            db = (1/self.batch_size) * np.sum(y_pred - y_batch)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        
        return self
    
    def predict(self, X):
        return X @ self.weights + self.bias


# Test
if __name__ == "__main__":
    X, y = make_regression(n_samples=100, n_features=3, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Standardize
    X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
    X_test = (X_test - X_train.mean(axis=0)) / X_train.std(axis=0)
    
    model = MiniBatchGradientDescent(learning_rate=0.1, n_iterations=1000, batch_size=32)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"Mini-Batch GD - MSE: {mean_squared_error(y_test, y_pred):.2f}, R²: {r2_score(y_test, y_pred):.3f}")





"""Important Notes:# 📚 **Complete Gradient Descent Cheat Sheet**

## 🎯 **LINEAR REGRESSION BASICS**

### **Normal Equation (Closed Form)**
```python
# Formula: θ = (X^T X)^(-1) X^T y
theta = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
```
- **One-step solution** - finds optimal weights mathematically
- **X_bias**: `np.c_[np.ones(n_samples), X]` adds bias column
- **theta[0]**: bias term, **theta[1:]**: weights
- **Weights order** matches feature columns order exactly

### **Prediction Formula**
```python
y_pred = X @ weights + bias
# Features × Weights + Bias
```

---

## 🔄 **GRADIENT DESCENT VARIENTS**

### **1. BATCH GRADIENT DESCENT**
```python
# Uses ALL training data in every iteration
for i in range(n_iterations):
    y_pred = X @ weights + bias           # Predict ALL
    dw = (1/n) * X.T @ (y_pred - y)       # Gradients from ALL
    db = (1/n) * np.sum(y_pred - y)       # Gradients from ALL
```

**Characteristics:**
- ✅ **Stable convergence** - smooth path to minimum
- ✅ **Guaranteed convergence** to global minimum (convex problems)
- ❌ **Memory intensive** - needs all data in memory
- ❌ **Slow per iteration** - processes all samples each time
- ❌ **Gets stuck** in local minima for non-convex problems

**Time Complexity:** `O(T × n × d)` per training

---

### **2. STOCHASTIC GRADIENT DESCENT (SGD)**
```python
# Uses ONE random sample per iteration
for i in range(n_iterations):
    idx = np.random.randint(0, n_samples)
    xi = X[idx:idx+1]  # Keep 2D shape!
    yi = y[idx]
    
    y_pred = xi @ weights + bias
    dw = xi.T @ (y_pred - yi)  # No averaging (single sample)
```

**Characteristics:**
- ✅ **Fast per iteration** - processes one sample
- ✅ **Memory efficient** - only one sample needed
- ✅ **Escapes local minima** - randomness helps
- ❌ **Noisy convergence** - zigzag path to minimum
- ❌ **May never converge** - keeps oscillating near minimum
- ❌ **Need more iterations** for same coverage

**Time Complexity:** `O(T × d)` but need `T × n` iterations for fair comparison

---

### **3. MINI-BATCH GRADIENT DESCENT**
```python
# Uses random BATCH of samples per iteration
for i in range(n_iterations):
    indices = np.random.choice(n_samples, batch_size, replace=False)
    X_batch = X[indices]    # Shape: (batch_size, n_features)
    y_batch = y[indices]
    
    y_pred = X_batch @ weights + bias
    dw = (1/batch_size) * X_batch.T @ (y_pred - y_batch)
```

**Characteristics:**
- ✅ **Balanced approach** - best of both worlds
- ✅ **GPU friendly** - parallelizable matrix operations
- ✅ **Good convergence** - less noisy than SGD
- ✅ **Faster than Batch GD** - processes subsets
- ❌ **Need to tune batch_size** - hyperparameter
- ❌ **Slightly more memory** than SGD

**Common Batch Sizes:** 32, 64, 128, 256

**Time Complexity:** `O(T × b × d)` where b = batch_size

---

## 📊 **SHAPES AND DIMENSIONS**

### **Key Shape Rules:**
```python
X.shape = (n_samples, n_features)     # e.g., (80, 3)
weights.shape = (n_features,)         # e.g., (3,) - CONSTANT!
bias = scalar                         # e.g., 50000

# Predictions:
X @ weights + bias → shape: (n_samples,)

# Batch selection:
X_batch = X[indices] → shape: (batch_size, n_features)

# Keep 2D shape: Use X[idx:idx+1] NOT X[idx]
```

### **Matrix Multiplication Shapes:**
```
(16, 3) @ (3,)    = (16,)     # Predictions
(3, 16) @ (16,)   = (3,)      # Gradient calculation
(3,)    - (3,)    = (3,)      # Weight update
```

---

## **HYPERPARAMETERS**

### **Learning Rate (lr)**
- **Too small** (0.001): Very slow convergence
- **Too large** (1.0): Overshooting, divergence
- **Good range**: 0.01 - 0.1
- **Adaptive**: Can decrease over time

### **Iterations vs Epochs**
- **Iteration**: One update step
- **Epoch**: One pass through entire dataset
- **Batch GD**: 1 iteration = 1 epoch
- **SGD**: n iterations = 1 epoch
- **Mini-batch**: (n/batch_size) iterations = 1 epoch

---

## **EVALUATION METRICS**

### **Mean Squared Error (MSE)**
```python
MSE = (1/n) × Σ(actual - predicted)²
```
- **Measures**: Average squared errors
- **Penalizes** large errors heavily
- **Sensitive to outliers**
- **Units**: squared (hard to interpret directly)

### **R² Score (R-Squared)**
```python
R² = 1 - (Sum of squared errors) / (Total variance)
```
- **Measures**: Percentage of variance explained
- **Range**: -∞ to 1.0
- **1.0** = Perfect predictions
- **0.0** = No better than predicting mean
- **< 0** = Worse than predicting mean

**Interpretation:**
- 0.9 = Model explains 90% of variation
- 0.7 = Model explains 70% of variation
- 0.5 = Model explains 50% of variation

---

## **PRACTICAL TIPS**

### **Data Preprocessing:**
```python
# Always standardize for Gradient Descent
X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
X_test = (X_test - X_train.mean(axis=0)) / X_train.std(axis=0)
```

### **Convergence Detection:**
```python
# Check if gradients are near zero
if np.allclose(gradient, 0, atol=1e-6):
    print("Converged!")
```

### **Random State:**
```python
# For reproducible results
np.random.seed(42)
make_regression(..., random_state=42)
train_test_split(..., random_state=42)
```

---

## 🎯 **WHEN TO USE WHICH**

### **Batch Gradient Descent:**
- Small datasets that fit in memory
- When you want precise convergence
- Academic examples and learning
- Stable, deterministic results

### **Stochastic Gradient Descent:**
- Very large datasets (millions of samples)
- Online learning (streaming data)
- When you want fast iterations
- Can escape local minima

### **Mini-Batch Gradient Descent:**
- **Most practical scenarios**
- Deep learning (standard choice)
- When you have GPU acceleration
- Good balance of speed and stability

---

## ⚡ **PERFORMANCE COMPARISON**

| Metric | Batch GD | SGD | Mini-Batch GD |
|--------|----------|-----|---------------|
| **Convergence** | Smooth | Noisy | Balanced |
| **Speed/Iteration** | Slow | Fast | Medium |
| **Memory** | High | Low | Medium |
| **GPU Usage** | Good | Poor | Excellent |
| **Final Accuracy** | High | Variable | High |

---

## 🔥 **KEY INSIGHTS**

1. **Weights shape is constant** - doesn't change during training
2. **Features order matters** - weights match feature columns order
3. **Batch size affects** convergence speed and quality
4. **Learning rate is crucial** - too high/low breaks training
5. **Standardization helps** gradient descent converge faster
6. **All methods have same** asymptotic time complexity for equal epochs
7. **Mini-batch is usually** the best practical choice

This covers everything from basic linear regression to advanced gradient descent variants! """