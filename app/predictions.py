import streamlit as st
import numpy as np
import pandas as pd

from config import Config as c


prediction_columns = ["MLR Without Genetic", "MLR With Genetic"]
# error_columns = ["Error MSE MLR", "Error MSE MLR+Genetic", "Error RMSE MLR", "Error RMSE MLR+Genetic"]


def predict_future_v3(X, model, colname, scaler_y, shift):
    X = X.iloc[-shift:]
    pred = model.predict(X)
    pred = scaler_y.inverse_transform(pred)
    df = pd.DataFrame({colname: np.squeeze(pred)}, index=X.index + pd.Timedelta(days=shift))
    return df


def combine_predictions(rekap, prediksi_lanjut, prediksi_lanjut_ga):
    prediksi_lanjut_gabungan = pd.concat([prediksi_lanjut, prediksi_lanjut_ga], axis=1)
    dates_str = [date.strftime("%Y-%m-%d") for date in prediksi_lanjut_gabungan.index]
    prediksi_lanjut_gabungan.index = dates_str
    prediksi_df = rekap.append(prediksi_lanjut_gabungan)
    return prediksi_df


def predict_ranged_days(rekap, period, X_unshifted, model, model_ga, scaler_y):
    # Copy rekap
    rekap = rekap[prediction_columns].copy()

    # Berapa banyak data yang digeser
    shift = c.SHIFT
    
    # Dapatkan hasil prediksi pada masa depan
    prediksi_lanjut = predict_future_v3(
        X=X_unshifted, 
        model=model,
        colname="MLR Without Genetic",
        scaler_y=scaler_y,
        shift=shift
    )
    prediksi_lanjut_ga = predict_future_v3(
        X=X_unshifted, 
        model=model_ga,
        colname="MLR With Genetic",
        scaler_y=scaler_y,
        shift=shift
    )

    # Gabungkan hasil prediksi masa depan
    prediksi_df = combine_predictions(rekap, prediksi_lanjut, prediksi_lanjut_ga)

    # DataFrame pada waktu tertentu
    rekap_len = len(rekap)
    prediksi_tertentu_df = prediksi_df.iloc[rekap_len: rekap_len + period]
    return prediksi_tertentu_df
    



def prediction_date_based(date, X, model, model_ga, scaler_y):
    # Berapa banyak data yang digeser
    shift = c.SHIFT

    # Copy Dataframe
    pd_date = pd.to_datetime(date, format="%Y-%m-%d")
    start = pd_date - pd.Timedelta(days=shift)
    end = start

    X = X.copy()
    X = X.loc[start: end]
    
    predictions = model.predict(X)
    predictions_ga = model_ga.predict(X)

    predictions = scaler_y.inverse_transform(predictions)
    predictions_ga = scaler_y.inverse_transform(predictions_ga)

    predictions_data = {
        "MLR Without GA": np.squeeze(predictions), 
        "MLR With Genetic": np.squeeze(predictions_ga)
    }

    df = pd.DataFrame(
        data=predictions_data, 
        index=[pd_date]
    )
    
    return df