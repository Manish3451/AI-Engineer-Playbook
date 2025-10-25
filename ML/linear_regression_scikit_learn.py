import numpy as np 
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.preprocessing import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def main():
    X, y = make_regression(n_samples=200, n_features=5, noise=15, random_state=4)
    feature_names = [f"feature_{i+1}" for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=feature_names)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_test_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)


    print(f"mse:{mse}")
    print(f"mae:{mae}")
    print(f"r2:{r2}")



if __name__ == "__main__":
    main()