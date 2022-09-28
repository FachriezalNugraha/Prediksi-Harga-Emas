import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def rekap_table(X_test, y_test, model, model_ga, scaler_y):
    y_test_series = y_test.values
    predictions_series = model.predict(X_test)
    predictions_ga_series = model_ga.predict(X_test)

    y_test_series = np.squeeze(scaler_y.inverse_transform(y_test_series))
    predictions_series = np.squeeze(scaler_y.inverse_transform(predictions_series))
    predictions_ga_series = np.squeeze(scaler_y.inverse_transform(predictions_ga_series))

    index = [date.strftime("%Y-%m-%d") for date in y_test.index]

    rekap = pd.DataFrame({
        "Y_test": y_test_series,
        "MLR Without Genetic": predictions_series,
        "MLR With Genetic": predictions_ga_series,
        "Error MLR": abs(y_test_series - predictions_series),
        "Error MLR+Genetic": abs(y_test_series - predictions_ga_series),
        "Error MSE MLR": (y_test_series - predictions_series)**2,
        "Error MSE MLR+Genetic": (y_test_series - predictions_ga_series)**2,
        "Error RMSE MLR": np.sqrt((y_test_series - predictions_series)**2),
        "Error RMSE MLR+Genetic": np.sqrt((y_test_series - predictions_ga_series)**2),
        "MAPE MLR": abs((y_test_series - predictions_series) / y_test_series)*100%,
        "MAPE MLR+Genetic": abs((y_test_series - predictions_ga_series) / y_test_series)*100%,
    }, index=index)

    return rekap


def compar_error(rekap):
    mean_mse_error = rekap["Error MSE MLR"].mean()
    mean_rmse_error = np.sqrt(mean_mse_error)
    mean_mape = rekap["MAPE MLR"].mean()

    mean_ga_mse_error = rekap["Error MSE MLR+Genetic"].mean()
    mean_ga_rmse_error = np.sqrt(mean_ga_mse_error)
    mean_ga_mape = rekap["MAPE MLR+Genetic"].mean()

    return {
        "Rata-rata error MSE tanpa algoritma genetika": mean_mse_error, 
        "Rata-rata error MSE dengan algoritma genetika": mean_ga_mse_error, 
        "Rata-rata error RMSE tanpa algoritma genetika": mean_rmse_error, 
        "Rata-rata error RMSE dengan algoritma genetika": mean_ga_rmse_error,
        "Rata-rata MAPE tanpa algoritma genetika": mean_mape,
        "Rata-rata MAPE dengan algoritma genetika": mean_ga_mape,
    }


def compar_error_plain(mse, mse_ga, rmse, rmse_ga):
    return {
        "Rata-rata error MSE tanpa algoritma genetika": mse, 
        "Rata-rata error MSE dengan algoritma genetika": mse_ga, 
        "Rata-rata error RMSE tanpa algoritma genetika": rmse, 
        "Rata-rata error RMSE dengan algoritma genetika": rmse_ga
    }