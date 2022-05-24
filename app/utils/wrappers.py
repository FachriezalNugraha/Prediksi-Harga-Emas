import streamlit as st

def wrap_view(title):
    def decorate(func):
        def content_view(*args, **kwargs):
            with st.expander(title):
                func(*args, **kwargs)
        return content_view
    return decorate


def is_trained(func):
    def decorate(*args, **kwargs):
        if st.session_state.get("linreg_beli"):
            func(*args, **kwargs)
        else:
            st.write("Model belum dilatih.")

    return decorate