import numpy as np 
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)



ridge = Ridge(alpha=0.1)
ridge.fit(X_train_scaled, Y_train)

y_pred = ridge.predict(X_test_scaled)


print("Ridge Regression Results:")
print(f"Coefficients: {ridge.coef_}")
print(f"Intercept: {ridge.intercept_:.2f}")
print(f"R² Score: {ridge.score(X_test_scaled, y_test):.3f}")