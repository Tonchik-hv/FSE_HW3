from useful_package import polynom_3, hyperbola
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
import pandas as pd

def main():
    X = pd.DataFrame([i for i in range(1, 200)])
    y_polynom = pd.DataFrame([polynom_3(i) for i in range(1, 200)])
    y_hyperbola = pd.DataFrame([hyperbola(i) for i in range(1, 200)])

    X_train_polynom, X_test_polynom, y_train_polynom, y_test_polynom = train_test_split(X, y_polynom, test_size = 0.2)
    X_train_hyperbola, X_test_hyperbola, y_train_hyperbola, y_test_hyperbola = train_test_split(X, y_polynom, test_size = 0.2)

    RF_regressor = RandomForestRegressor(n_estimators = 10)
    RF_regressor.fit(X_train_polynom, y_train_polynom)

    y_pred_polynom = RF_regressor.predict(X_test_polynom)

    MSE_polynom = mse(y_test_polynom, y_pred_polynom.ravel())

    RF_regressor = RandomForestRegressor(n_estimators = 10)
    RF_regressor.fit(X_train_hyperbola, y_train_hyperbola)

    y_pred_hyperbola = RF_regressor.predict(X_test_hyperbola)

    MSE_hyperbola = mse(y_test_hyperbola, y_pred_hyperbola.ravel())
    
    return MSE_polynom, MSE_hyperbola

    


if __name__ == "__main__":
    main()
