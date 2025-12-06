import streamlit as st
import streamlit.components.v1 as components

st.title("Evidently Report Viewer")

report_file = "/app/reports/report.html"

try:
    with open(report_file, "r") as f:
        html_content = f.read()
    components.html(html_content, height=800, width=1000, scrolling=True)
except FileNotFoundError:
    st.warning("No report found. Please train a model first by calling the training API.")
    st.info("POST to http://localhost:8080/")

