import streamlit as st
import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import Legend
from bokeh.palettes import Category10_3, Category10_4
from bokeh.transform import dodge
from src.models import prediction_columns


BG_COLOR = "#0E1117"
TEXT_COLOR = "#FAFAFA"

def apply_bokeh_dark(p):
    # Outline
    p.outline_line_color = None

    # Grid
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    # Format Text
    p.title.align = "center"
    p.title.text_font_size = "14px"

    # Background Color
    p.background_fill_color = BG_COLOR
    p.border_fill_color = BG_COLOR
    p.legend.background_fill_color = BG_COLOR

    # Text Color
    p.title.text_color = TEXT_COLOR
    p.legend.label_text_color = TEXT_COLOR
    p.xaxis.major_label_text_color = TEXT_COLOR
    p.yaxis.major_label_text_color = TEXT_COLOR

    # Axes Line Color
    p.xaxis.major_tick_line_color = TEXT_COLOR
    p.xaxis.axis_line_color = TEXT_COLOR
    p.yaxis.major_tick_line_color = TEXT_COLOR
    p.yaxis.axis_line_color = TEXT_COLOR


def compar_table(X_test, y_test, model, model_ga, mode):
    y_test_series = y_test.values
    predictions_series = model.predict(X_test)
    predictions_ga_series = model_ga.predict(X_test)

    scaler = st.session_state["scaler_{}_y".format(mode)]
    y_test_series = np.squeeze(scaler.inverse_transform(y_test_series))
    predictions_series = np.squeeze(scaler.inverse_transform(predictions_series))
    predictions_ga_series = np.squeeze(scaler.inverse_transform(predictions_ga_series))

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
    }, index=index)

    rekap_show = rekap.sort_index().style.format(precision=2)
    return rekap, rekap_show


def compar_error(rekap):
    mean_mse_error = rekap["Error MSE MLR"].mean()
    mean_rmse_error = np.sqrt(mean_mse_error)
    mean_ga_mse_error = rekap["Error MSE MLR+Genetic"].mean()
    mean_ga_rmse_error = np.sqrt(mean_ga_mse_error)

    return {
        "Rata-rata error MSE tanpa algoritma genetika": mean_mse_error, 
        "Rata-rata error MSE dengan algoritma genetika": mean_ga_mse_error, 
        "Rata-rata error RMSE tanpa algoritma genetika": mean_rmse_error, 
        "Rata-rata error RMSE dengan algoritma genetika": mean_ga_rmse_error
    }


def error_bar_chart(rekap, days=30):
    different = rekap[["Error MSE MLR", "Error MSE MLR+Genetic"]].iloc[-days:]
    dates_str = list(different.index)
    different["date"] = dates_str

    p = figure(width=900, height=500, x_range=dates_str, tools=[], sizing_mode="stretch_width")

    v1 = p.vbar(x=dodge("date", -0.11, range=p.x_range), width=0.2, top="Error MSE MLR", 
                color=Category10_4[0], source=different)

    v2 = p.vbar(x=dodge("date", 0.11, range=p.x_range), width=0.2, top="Error MSE MLR+Genetic", 
                color=Category10_4[1], source=different)

    legend = Legend(items=[
        ("Error MSE MLR", [v1]), 
        ("Error MSE MLR+Genetic", [v2])
    ], location="left", title_text_color=TEXT_COLOR, label_text_color=TEXT_COLOR,
    background_fill_color=BG_COLOR)

    p.xaxis.major_label_orientation = "vertical"
    
    apply_bokeh_dark(p)
    p.add_layout(legend, "above")

    return p


def error_line_chart(rekap, days=30):
    different = rekap[["Error MSE MLR", "Error MSE MLR+Genetic"]].iloc[-days:]
    dates_str = list(different.index)
    different["date"] = dates_str
    p = figure(width=900, height=500, x_range=dates_str, sizing_mode="stretch_width")

    l1 = p.line(x="date", y="Error MSE MLR", line_width=1.5, color=Category10_4[0], source=different)
    c1 = p.circle(x="date", y="Error MSE MLR", size=5, color=Category10_4[0], source=different)
    l2 = p.line(x="date", y="Error MSE MLR+Genetic", line_width=1.5, color=Category10_4[1], source=different)
    c2 = p.circle(x="date", y="Error MSE MLR+Genetic", size=5, color=Category10_4[1], source=different)

    legend = Legend(items=[
        ("Error MSE MLR", [l1, c1]), 
        ("Error MSE MLR+Genetic", [l2, c2])
    ], location="left", title_text_color=TEXT_COLOR, label_text_color=TEXT_COLOR,
    background_fill_color=BG_COLOR)

    p.xaxis.major_label_orientation = "vertical"
    apply_bokeh_dark(p)
    p.add_layout(legend, "above")

    return p


def predictions_line_chart(df):
    items = []
    dates_str = list(df.index)
    df["date"] = dates_str
    p = figure(width=900, height=600, x_range=dates_str, sizing_mode="stretch_width") 

    for name, color in zip(prediction_columns, Category10_3):
        line = p.line(x="date", y=name, line_width=1.5, color=color, source=df)
        scatter = p.scatter(x="date", y=name, size=3, color=color, source=df)
        p.xaxis.major_label_orientation = "vertical"
        p.legend.location = "top_left"
        items.append((name, [line, scatter]))

    # Tempelkan legenda
    p.add_layout(
        Legend(
            items=items, location="left", title_text_color=TEXT_COLOR, 
            label_text_color=TEXT_COLOR, background_fill_color=BG_COLOR)
        , "above")

    apply_bokeh_dark(p)

    return p