import streamlit as st

st.set_page_config(page_title="privacy-policy", page_icon="🔑")

st.markdown("# Privacy Policy")
st.sidebar.header("Privacy Policy")

with open("./privacy-policy.md", "r") as file:
    privacy_policy_content = file.read()
st.markdown(privacy_policy_content)