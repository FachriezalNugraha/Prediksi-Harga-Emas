import streamlit as st

from views import *
from styles import css_style

# Muat style CSS
css_style()

# Tampilan Antar Muka
view_home()
view_tutorial()
view_dataset_type()
view_parameter()
view_train()
view_result()
view_comparison()
view_charts()
view_predict_period()
view_predict_date()