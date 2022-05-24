from bokeh.plotting import figure
from bokeh.models import Legend
from bokeh.palettes import Category10_3, Category10_4
from bokeh.transform import dodge

from predictions import prediction_columns


BG_COLOR = "#0E1117"
TEXT_COLOR = "#FAFAFA"
color1, color2, color3, color4 = Category10_4

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


def error_bar_chart(rekap, days=30):
    different = rekap[["Error MSE MLR", "Error MSE MLR+Genetic"]].iloc[-days:]
    dates_str = list(different.index)
    different["date"] = dates_str

    p = figure(width=900, height=500, x_range=dates_str, tools=[], sizing_mode="stretch_width")

    v1 = p.vbar(x=dodge("date", -0.11, range=p.x_range), width=0.2, top="Error MSE MLR", 
                color=color1, source=different)

    v2 = p.vbar(x=dodge("date", 0.11, range=p.x_range), width=0.2, top="Error MSE MLR+Genetic", 
                color=color2, source=different)

    legend = Legend(
        items=[("Error MSE MLR", [v1]), ("Error MSE MLR+Genetic", [v2])], 
        location="left", 
        title_text_color=TEXT_COLOR, 
        label_text_color=TEXT_COLOR,
        background_fill_color=BG_COLOR
    )

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