import streamlit as st


def css_style():
    st.markdown("""
    <style>
        div.streamlit-expanderHeader {
            font-size: 2rem;
        }

        .epcbefy1 {
            border: none;
        }
    </style>
    """ ,unsafe_allow_html=True)