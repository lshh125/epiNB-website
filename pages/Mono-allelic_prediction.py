import streamlit as st
import epinb

st.markdown("# Mono-allelic prediction")

col1, col2 = st.columns(2)
with col1:
    training_data = st.text_area("Training peptides")
with col2:
    test_data = st.text_area("Candidate peptides")

col1, col2 = st.columns([1, 8])
with col1:
    if st.button('Run'):
        training_data = training_data.strip().split()
        test_data = test_data.strip().split()
        
        if len(training_data) < 10:
            raise ValueError("At least 10 training peptides required.")
            
        if len(test_data) < 1:
            raise ValueError("At least 1 andidate peptides required.")
        
        model = epinb.NBScore()
        model.fit(training_data)
        res = model.predict_details(test_data)
        with col2:
            st.download_button("Download results", res.to_csv(), "epiNB-predictions.csv", disabled=False)
        st.dataframe(res)
    else:
        with col2:
            st.download_button("Download results", "", disabled=True)
    
