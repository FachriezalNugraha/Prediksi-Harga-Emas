import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from config import Config as c


def convert_to_datetime(df, colname, format):
    df[colname] = pd.to_datetime(df[colname], format=format)
    return df

def reverse_df(df):
    df = df.iloc[::-1]
    df = df.reset_index(drop=True)
    return df

def scale_data(df):
    return df / 10000

def get_variables(df, mode, period=1):
    # Copy dataframe supaya tidak menimpa dataframe asli
    df = df.copy()

    # Ubah huruf pertama mode menjadi kapital
    mode = mode.title()

    # Tentukan variabel
    predictor = df[["Inflasi", "HargaMinyak", f"Kurs{mode}"]]
    respon = df[[f"Harga{mode}"]]

    # Mundurkan variabel respon ke observasi pada beberapa hari yang lalu
    respon = respon.shift(-period)

    # Hilangkan missing value
    respon = respon.dropna()

    # Simpan predictor asli
    st.session_state["predictor_{}".format(mode.lower())] = predictor
    predictor_unshifted = predictor.copy()

    # Sejajarkan variabel respon dan prediktor supaya memiliki jumlah observasi yang sama
    predictor = predictor.loc[respon.index]

    # Sesuaikan tanggal pada variabel respon
    respon.index += pd.Timedelta(days=period)

    return predictor, predictor_unshifted, respon


def prepare_data(df, mode):
    # Pre - Urutkan Data
    df = reverse_df(df)

    # Pre - Mengubah indeks menjadi tanggal
    df = convert_to_datetime(df=df, colname=c.DATE_COL, format=c.DATE_FORMAT)

    # Pre - Gunakan kolom tanggal sebagai indeks
    df = df.set_index(c.DATE_COL)

    # Pre - Ambil variabel dari Data
    X, X_real, y = get_variables(df, mode, period=c.SHIFT)

    # Pre - Atur Skala Data
    # y = scale_data(y)

    return X, X_real, y


def init_scaler(data, scaler):
    scaler_instance = scaler()
    scaler_fitted = scaler_instance.fit(data)
    return scaler_fitted

def apply_test_scaler(X_test, y_test, mode):
    X_test = X_test.copy()
    y_test = y_test.copy()

    X_test[:] = st.session_state["scaler_{}_x".format(mode)].transform(X_test)
    y_test[:] = st.session_state["scaler_{}_y".format(mode)].transform(y_test)

    return {"X_test": X_test, "y_test": y_test}


def date_offset(df):
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d") + pd.Timedelta(days=st.session_state["shift"])
    df.index = [date.strftime("%Y-%m-%d") for date in df.index]


def preprocess_data(X, X_unshifted, y, test_size):
    # Split Data
    ## Data acak
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    # Inisiasi Scaler
    scaler_X = init_scaler(X_train, MinMaxScaler)
    scaler_y = init_scaler(y_train, MinMaxScaler)

    # Terapkan Scaler
    X_train.loc[:] = scaler_X.transform(X_train)
    X_test.loc[:] = scaler_X.transform(X_test)
    X_unshifted.loc[:] = scaler_X.transform(X_unshifted)
    y_train.loc[:] = scaler_y.transform(y_train)
    y_test.loc[:] = scaler_y.transform(y_test)

    return X_train, X_test, X_unshifted, y_train, y_test, scaler_X, scaler_y


def sort_splitted_data(train_data, test_data):
    train_len = len(train_data)
    
    sorted_data = pd.concat([train_data, test_data]).sort_index()
    train_sorted, test_sorted = sorted_data[:train_len], sorted_data[train_len:]

    return train_sorted, test_sorted