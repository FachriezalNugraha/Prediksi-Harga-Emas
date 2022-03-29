import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.data import load_data, load_custom_data, verify_data
from src.models import gen_algo, evaluate, combine_predictions, prediction_date_based
from src.pre import preprocess_data, apply_test_scaler
from src.visualization import compar_table, compar_error, error_bar_chart, error_line_chart, predictions_line_chart


__all__ = [
    "view_home", "view_tutorial", "view_dataset_type", "view_parameter", "view_train",
    "view_result", "view_comparison", "view_charts", "view_predict_period", "view_predict_date"
]

# Inisiasi
df = load_data("app/data/dataset_full.csv")
session = st.session_state
MODES = ["beli", "jual"]


def wrap_view(title):
    def decorate(func):
        def content_view():
            with st.expander(title):
                func()
        return content_view
    return decorate


def is_trained(func):
    def decorate(*args, **kwargs):
        if session.get("linreg_beli"):
            func(*args, **kwargs)
        else:
            st.write("Model belum dilatih.")

    return decorate
    

def view_home():
    st.title("Prediksi Harga Emas")
    st.markdown("""
    ---
    Aplikasi untuk melakukan prediksi pada harga beli dan harga jual emas. Model machine learning
    yang digunakan adalah regresi linier dan regresi linier dengan optimalisasi algoritma genetika.
    Pelatihan model dilakukan dengan menggunakan dataset harga emas pada kurun waktu `1 Januari 2017`
    hingga `31 Juli 2021`.

    Adapun fitur-fitur yang terdapat pada aplikasi ini adalah:
    - Prediksi harga emas pada jangka waktu tertentu.
    - Prediksi harga emas pada tanggal tertentu.

    #
    """)


def view_tutorial():
    st.subheader("Cara Penggunaan")
    st.markdown("""
    Prediksi Harga Emas dapat dilakukan dengan langkah-langkah sebagai berikut:
    - Memilih dataset asli atau custom dari pengguna. 
    - Memasukkan parameter training dan algoritma genetika.
    - Melatih model dengan parameter yang ditentukan.
    - Mulai Prediksi.

    #
    """)


@wrap_view(title="Tipe Dataset")
def view_dataset_type():
    st.write("Pilih salah satu dari tipe dibawah")
    option = st.radio(label="Tipe Dataset", options=["Asli", "Custom"])

    if option == "Custom":
        st.markdown("""
        Pastikan bahwa data memiliki:
        - 2 kolom dengan nama `HargaBeli` dan `HargaJual` yang berisikan bilangan cacah / bulat
        - 1 kolom dengan nama `Date` yang berisikan tanggal dengan format `DD/MM/YYYY`
        - Jumlah data lebih dari 20
        #
        """)
        csv_file = st.file_uploader(label="Unggah file .csv, .xlsx", type=["csv", "xlsx"])
        if csv_file:
            custom_df = load_custom_data(csv_file)
            valid = verify_data(custom_df)
            session["dataset"] = custom_df if valid else df
            session["dataset_type"] = "Custom" if valid else "Asli"
    
    else:
        st.write("Anda akan menggunakan dataset asli dari server")
        session["dataset"] = df
        session["dataset_type"] = "Asli"


@wrap_view(title="Parameter")
def view_parameter():
    with st.form("Parameter"):
        st.write("Parameter data")
        test_size = st.number_input(label="Ukuran Data Test", min_value=0.1, max_value=0.5, step=0.05)
        st.markdown("")

        st.write("Parameter Algoritma Genetika")
        generation = st.number_input(label="Jumlah Generasi", min_value=10, step=10)
        size = st.number_input(label="Ukuran Populasi", min_value=100, step=100)
        cr = st.number_input(label="Crossover Rate", min_value=0.0, max_value=1.0, step=0.1)
        mr = st.number_input(label="Mutation Rate", min_value=0.0, max_value=1.0, step=0.1)


        is_submit = st.form_submit_button("Simpan")
    
    if is_submit:
        session["test"] = test_size
        session["n_gen"] = generation
        session["size"] = size
        session["cr"] = cr
        session["mr"] = mr

        beli_train, beli_test, jual_train, jual_test = preprocess_data(session["dataset"], session["test"])
        
        session["beli_train"] = beli_train
        session["beli_test"] = beli_test
        session["jual_train"] = jual_train
        session["jual_test"] = jual_test

        len_test = len(session["beli_test"]["X_test"])
        session["beli_real_test"] = {
            "X_test": session["X_beli"].iloc[-len_test:], 
            "y_test": session["y_beli"].iloc[-len_test:]
        }
        session["jual_real_test"] = {
            "X_test": session["X_jual"].iloc[-len_test:], 
            "y_test": session["y_jual"].iloc[-len_test:]
        }

        session["beli_real_test"] = apply_test_scaler(**session["beli_real_test"], mode="beli")
        session["jual_real_test"] = apply_test_scaler(**session["jual_real_test"], mode="jual")


@wrap_view("Latih Model")
def view_train():
    prerequisites = ["dataset_type", "test", "n_gen", "size", "cr", "mr"]
    data = [str(session.get(req, "-belum ditentukan-")) for req in prerequisites]
    index = ["Tipe Dataset", "Ukuran Data Test", "Jumlah Generasi", "Ukuran Populasi", "Crossover Rate", "Mutation Rate"]
    
    st.markdown("""
    Pastikan anda telah memilih tipe dataset dan parameter sebelum mulai melatih model.
    Jika sudah, tekan tombol `Latih` untuk mulai melatih model.
    """)
    st.table(pd.DataFrame(data, index=index, columns=["Nilai"]))

    is_train = st.button("Latih")
    st.markdown("#")

    if is_train and not session.get("test"):
        st.warning("Parameter belum ditentukan")

    elif is_train and session.get("test"):
        # Train Linreg
        linreg_beli = LinearRegression().fit(session["beli_train"]["X_train"], session["beli_train"]["y_train"])
        linreg_jual = LinearRegression().fit(session["jual_train"]["X_train"], session["jual_train"]["y_train"])

        # Train GA
        ga_input = dict(size=session["size"], 
                        n_gen=session["n_gen"], 
                        cr=session["cr"], 
                        mr=session["mr"])

        population_beli, fitness_beli, linreg_beli_ga = gen_algo(**ga_input, **session["beli_train"], mode="beli")
        population_jual, fitness_jual, linreg_jual_ga = gen_algo(**ga_input, **session["jual_train"], mode="jual")

        session["fitness_beli"] = fitness_beli[0]
        session["fitness_jual"] = fitness_jual[0]
        session["linreg_beli"] = linreg_beli
        session["linreg_jual"] = linreg_jual
        session["linreg_beli_ga"] = linreg_beli_ga
        session["linreg_jual_ga"] = linreg_jual_ga



def show_result_matric(mode):
    col1, col2 = st.columns([6, 6])

    with col1:
        st.write("Metrik regresi linier pada harga {}".format(mode))
        results = evaluate(session["linreg_{}".format(mode)], mode="{}".format(mode))
        
        for metric in results:
            st.write("{} : {:.3f}".format(metric.upper(), results[metric]))
        st.write("--")
        st.markdown("---")

    with col2:
        st.write("Metrik regresi linier + GA pada harga {}".format(mode))
        results = evaluate(session["linreg_{}_ga".format(mode)], mode="{}".format(mode))
        
        for metric in results:
            st.write("{} : {:.3f}".format(metric.upper(), results[metric]))
        st.write("Fitness {} : {:.4f}".format(mode, session["fitness_{}".format(mode)]))
        st.markdown("---")


@wrap_view("Hasil Evaluasi Model")
@is_trained
def view_result():
    show_mode = st.selectbox("Jenis", ["Harga Jual", "Harga Beli", "Semua"], key="result")
    show_mode = show_mode.lower().split()
    st.markdown("#")
    
    for mode in MODES:
        if mode in show_mode or "semua" in show_mode:
            show_result_matric(mode)
        else:
            continue



def show_pred_comparison(mode):
    st.markdown("**Prediksi pada harga {}**".format(mode))
    rekap, rekap_show = compar_table(
        model=session["linreg_{}".format(mode)], 
        model_ga=session["linreg_{}_ga".format(mode)],
        mode=mode,
        **session["{}_real_test".format(mode)]
    )
    session["rekap_{}".format(mode)] = rekap
    st.dataframe(rekap_show)

    errors = compar_error(rekap)
    for label in errors:
        st.write("{} : {:.3f}".format(label, errors[label]))

    st.markdown("#")


@wrap_view(title="Perbandingan Prediksi")
@is_trained
def view_comparison():
    show_mode = st.selectbox("Jenis", [ "Semua", "Harga Jual", "Harga Beli"], key="comparison")
    st.markdown("#")
    show_mode = show_mode.lower().split()
        
    for mode in MODES:
        if mode in show_mode or "semua" in show_mode:
            show_pred_comparison(mode)
        else:
            continue
        

def show_chart(mode):
    chart_functs = [("Batang", error_bar_chart), ("Garis", error_line_chart)]
    for shape, func in chart_functs:
        st.markdown("**Diagram {} MSE pada harga {}**".format(shape, mode))
        rekap = session["rekap_{}".format(mode)]
        chart = func(rekap=rekap)
        st.bokeh_chart(chart)
    st.markdown("#")


@wrap_view("Visualisasi Error")
@is_trained
def view_charts():
    show_mode = st.selectbox("Jenis", [ "Semua", "Harga Jual", "Harga Beli"], key="charts")
    st.markdown("#")
    show_mode = show_mode.lower().split()
    
    for mode in MODES:
        if mode in show_mode or "semua" in show_mode:
            show_chart(mode)
        else:
            continue
            


def show_predict_period(mode, period):
    predict_period = combine_predictions(
        period=session["period"], 
        X_test=session["predictor_{}".format(mode)], 
        rekap=session["rekap_{}".format(mode)],
        model=session["linreg_{}".format(mode)],
        model_ga=session["linreg_{}_ga".format(mode)],
        mode=mode
    )
    value_chart = predictions_line_chart(predict_period)
    error_chart = error_bar_chart(predict_period, days=session["period"] * 2)

    st.markdown("**Tabel prediksi harga {} pada jangka waktu {} hari**".format(mode, period))
    st.dataframe(predict_period.style.format(precision=2))
    st.markdown("##")
    st.markdown("**Diagram garis harga {} pada jangka waktu {} hari**".format(mode, period))
    st.bokeh_chart(value_chart)
    st.markdown("**Diagram error {} pada jangka waktu {} hari**".format(mode, period))
    st.bokeh_chart(error_chart)
    st.markdown("#")


@wrap_view("Prediksi Jangka Waktu Tertentu")
@is_trained
def view_predict_period():
    with st.form("Period"):
        show_mode = st.selectbox("Jenis", ["Semua", "Harga Jual", "Harga Beli"], key="result")
        show_mode = show_mode.lower().split()
        period = st.number_input(label="Jangka Waktu Prediksi (hari)", min_value=5, max_value=30)
        is_submit = st.form_submit_button("Prediksi")
    st.markdown("#")
    
    if is_submit:
        session["period"] = period
    else:
        return
    
    for mode in MODES:
        if mode in show_mode or "semua" in show_mode:
            show_predict_period(mode, period)
        else:
            continue

        

def show_predict_date(mode, date):
    shift = session["shift"]
    predictions_date = prediction_date_based(
        date=date, 
        X=session["predictor_{}".format(mode)].iloc[:-shift],
        model=st.session_state["linreg_beli"],
        model_ga=st.session_state["linreg_beli_ga"],
        mode=mode
    )

    st.write("Prediksi harga {} emas pada {:%d %B %Y}".format(mode, date))
    st.dataframe(predictions_date.style.format(precision=2))


@wrap_view("Prediksi Tanggal Tertentu")
@is_trained
def view_predict_date():
    shift = session["shift"]
    min_value=session["predictor_beli"].index[shift]
    max_value=session["predictor_beli"].index[-1] + pd.Timedelta(days=shift)

    with st.form("Date"):
        show_mode = st.selectbox("Jenis", ["Semua", "Harga Jual", "Harga Beli"], key="result")
        date = st.date_input(label="Masukkan Tanggal", value=min_value, min_value=min_value, max_value=max_value)
        is_submit = st.form_submit_button("Prediksi")
        show_mode = show_mode.lower().split()
    
    if is_submit:
        for mode in MODES:
            if mode in show_mode or "semua" in show_mode:
                show_predict_date(mode, date)
            else:
                continue