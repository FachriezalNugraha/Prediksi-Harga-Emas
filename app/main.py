import streamlit as st

from views import main
from styles import css_style

# Ubah tampilan menjadi lebar
st.set_page_config(layout="wide")

# Muat style CSS
css_style()

# Tampilan Antar Muka
main()
