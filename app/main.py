import streamlit as st

from views import main
from styles import css_style

# Ubah tampilan menjadi lebar
st.set_page_config(layout="wide")

# Muat style CSS
css_style()

# Tampilan Antar Muka
main()
# view_home()
# view_tutorial()
# view_dataset_type()
# view_data_parameter()
# view_gen_parameter()
# view_train()
# view_result()
# view_comparison()
# view_charts()
# view_predict_period()
# view_predict_date()