import streamlit as st

st.title("Welcome to epiNB online app")
st.markdown("**E**fficient, **p**recise and **i**nterpretable HLA-I binding **epi**tope identification based on **N**aive **B**ayes formulation")

st.header("Use cases → submodules")
st.markdown("""- I have some candidate peptides and I want to know if they will present on one or more HLAs → Library-based prediction."
- I have some peptides from one allele and I want to use them to identify more → Mono-allelic prediction.
- I know patient HLA alleles, have some peptides from the patient but don't know which specific allele each one is from, and I want to use them to predict neoantigens → Semi-supervised patient prediction.""")
st.markdown("Please use the sidebar on the left to navigate the submodules.")
