import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split


@st.cache
def sort_by_column(df, col_name="Date"):
    # Membalikkan urutan
    df = df.iloc[::-1]
    df = df.reset_index(drop=True)

    # Mengubah indeks menjadi tanggal
    df[col_name] = pd.to_datetime(df[col_name], format="%d/%m/%Y")
    df = df.set_index(col_name)

    return df


@st.cache
def scale_data(df):
    return df / 1000


@st.cache
def get_variables(df, column, period=1):
    # Copy dataframe supaya tidak menimpa dataframe asli
    df = df.copy()

    # Buat variabel prediktor berdasarkan banyaknya periode waktu
    response = df[[column]]
    predictor = pd.concat([response.shift(i) for i in range(1, period + 1)], axis=1)

    # Sunting nama kolom supaya mudah dibaca
    response.columns = [f"{column}"]
    predictor.columns = [f"{column}-{lag}" for lag in range(1, period + 1)]

    # Gabungkan respon dan prediktor
    variables = pd.concat([response, predictor], axis=1)

    # Hilangkan nilai NaN
    variables = variables.dropna()

    # Kembalikan prediktor dan respon
    return variables[predictor.columns], variables[response.columns]


@st.cache
def split_data(X, y, test_size, shuffle=False):
    return train_test_split(X, y, test_size=test_size, shuffle=shuffle)


@st.cache
def preprocess_data(df, test_size):
    # Pre - Urutkan Data
    df_sorted = sort_by_column(df)

    # Pre - Variabel dari Data
    X_beli, y_beli = get_variables(df_sorted, "HargaBeli", period=4)
    X_jual, y_jual = get_variables(df_sorted, "HargaJual", period=4)

    # Pre - Atur Skala Data
    X_beli = scale_data(X_beli)
    y_beli = scale_data(y_beli)
    X_jual = scale_data(X_jual)
    y_jual = scale_data(y_jual)

    # Split Data Harga Beli
    X_beli_train, X_beli_test, y_beli_train, y_beli_test = split_data(X_beli, y_beli, test_size=test_size)

    # Split Data Harga Jual
    X_jual_train, X_jual_test, y_jual_train, y_jual_test = split_data(X_jual, y_jual, test_size=test_size)

    # Serialisasi data
    beli_train = {"X_train": X_beli_train, "y_train": y_beli_train}
    beli_test = {"X_test": X_beli_test, "y_test": y_beli_test}
    jual_train = {"X_train": X_jual_train, "y_train": y_jual_train}
    jual_test = {"X_test": X_jual_test, "y_test": y_jual_test}

    return beli_train, beli_test, jual_train, jual_test