import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


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


def get_variables(df, mode, period=1):
    # Copy dataframe supaya tidak menimpa dataframe asli
    df = df.copy()
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

    # Sejajarkan variabel respon dan prediktor supaya memiliki jumlah observasi yang sama
    predictor = predictor.loc[respon.index]

    # Sesuaikan tanggal pada variabel respon
    respon.index += pd.Timedelta(days=60)

    return predictor, respon


def split_data(X, y, test_size):
    return train_test_split(X, y, test_size=test_size, random_state=42)


def preprocess_data(df, test_size):
    # Pre - Urutkan Data
    df_sorted = sort_by_column(df)

    # Pre - Variabel dari Data
    shift = 60
    X_beli, y_beli = get_variables(df_sorted, "beli", period=shift)
    X_jual, y_jual = get_variables(df_sorted, "jual", period=shift)

    # Pre - Atur Skala Data
    # X_beli = scale_data(X_beli)
    y_beli = scale_data(y_beli)
    # X_jual = scale_data(X_jual)
    y_jual = scale_data(y_jual)

    st.session_state["shift"] = shift
    st.session_state["X_beli"] = X_beli
    st.session_state["y_beli"] = y_beli
    st.session_state["X_jual"] = X_jual
    st.session_state["y_jual"] = y_jual

    

    # Split Data Harga Beli
    X_beli_train, X_beli_test, y_beli_train, y_beli_test = split_data(X_beli, y_beli, test_size=test_size)

    # Split Data Harga Jual
    X_jual_train, X_jual_test, y_jual_train, y_jual_test = split_data(X_jual, y_jual, test_size=test_size)

    # Normalisasi Data
    scaler_beli_x = MinMaxScaler().fit(X_beli_train)
    scaler_beli_y = MinMaxScaler().fit(y_beli_train)
    scaler_jual_x = MinMaxScaler().fit(X_jual_train)
    scaler_jual_y = MinMaxScaler().fit(y_jual_train)

    st.session_state["scaler_beli_x"] = scaler_beli_x
    st.session_state["scaler_beli_y"] = scaler_beli_y
    st.session_state["scaler_jual_x"] = scaler_jual_x
    st.session_state["scaler_jual_y"] = scaler_jual_y

    X_beli_train[:] = scaler_beli_x.transform(X_beli_train)
    X_beli_test[:] = scaler_beli_x.transform(X_beli_test)
    y_beli_train[:] = scaler_beli_y.transform(y_beli_train)
    y_beli_test[:] = scaler_beli_y.transform(y_beli_test)

    X_jual_train[:] = scaler_jual_x.transform(X_jual_train)
    X_jual_test[:] = scaler_jual_x.transform(X_jual_test)
    y_jual_train[:] = scaler_jual_y.transform(y_jual_train)
    y_jual_test[:] = scaler_jual_y.transform(y_jual_test)

    st.session_state["predictor_beli"][:] = scaler_beli_x.transform(st.session_state["predictor_beli"])
    st.session_state["predictor_jual"][:] = scaler_jual_x.transform(st.session_state["predictor_jual"])

    # Serialisasi data
    beli_train = {"X_train": X_beli_train, "y_train": y_beli_train}
    beli_test = {"X_test": X_beli_test, "y_test": y_beli_test}
    jual_train = {"X_train": X_jual_train, "y_train": y_jual_train}
    jual_test = {"X_test": X_jual_test, "y_test": y_jual_test}

    return beli_train, beli_test, jual_train, jual_test


def apply_test_scaler(X_test, y_test, mode):
    X_test = X_test.copy()
    y_test = y_test.copy()

    X_test[:] = st.session_state["scaler_{}_x".format(mode)].transform(X_test)
    y_test[:] = st.session_state["scaler_{}_y".format(mode)].transform(y_test)

    return {"X_test": X_test, "y_test": y_test}

def date_offset(df):
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d") + pd.Timedelta(days=st.session_state["shift"])
    df.index = [date.strftime("%Y-%m-%d") for date in df.index]
