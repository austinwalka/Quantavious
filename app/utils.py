# utils.py
import webbrowser
import streamlit as st

def trigger_colab_retrain(colab_url: str):
    """
    Opens Colab notebook to manually retrain stock predictions.
    """
    webbrowser.open_new_tab(colab_url)
    st.info("Colab notebook opened. Run the cells to retrain the stock predictions.")
