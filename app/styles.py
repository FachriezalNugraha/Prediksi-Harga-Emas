import streamlit as st


def css_style():
    st.markdown("""
    <style>
        div.streamlit-expanderHeader {
            font-size: 2rem;
        }
    </style>
    """ ,unsafe_allow_html=True)