import streamlit as st

st.markdown("# Mono-allelic prediction")

col1, col2 = st.columns(2)
with col1:
    training_data = st.text_area("Training peptides")
with col2:
    test_data = st.text_area("Candidate peptides")

col1, col2 = st.columns([1, 8])
with col1:
    if st.button('Run'):
        res = test_data
        with col2:
            st.download_button("Download results", res, disabled=False)
        st.write(res)
    else:
        with col2:
            st.download_button("Download results", "", disabled=True)
    

