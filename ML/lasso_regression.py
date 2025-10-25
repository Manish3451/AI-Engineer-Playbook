import numpy as np 
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso



X, y = make_regression(n_samples=1000, n_features=5, noise=10, random_state=42)

X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.2,random_state=42)


scaler = StandardScaler()

X_train_scaled = scaler(X_train)
X_test_scaled = scaler(X_test)


lasso = Lasso(alpha=1.0)
lasso.fit(X_train_scaled, Y_train)

y_pred = lasso.predict(X_test_scaled)

print("Lasso Regression Results:")
print(f"Coefficients: {lasso.coef_}")
print(f"Intercept: {lasso.intercept_:.2f}")
print(f"R² Score: {lasso.score(X_test_scaled, y_test):.3f}")
print(f"Features selected: {np.sum(lasso.coef_ != 0)}/{len(lasso.coef_)}")