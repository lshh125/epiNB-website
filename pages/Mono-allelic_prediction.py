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
        # st.dataframe(res)
    else:
        with col2:
            st.download_button("Download results", "", disabled=True)

with st.expander("Help"):
    st.markdown("1. Copy your training peptides into the left text box.")
    st.markdown("2. Copy your candidate peptides to be classified into the right text box.")
    st.markdown("3. Click the \"Run\" button.")
    st.markdown("4. Wait till the \"Download results\" button become valid, and click it to download the results.")
    
with st.expander("Want some sample input to try out?"):
    st.markdown("We used data for A0203 as an example here.")
    st.markdown("You can download sample [training](https://raw.githubusercontent.com/lshh125/epiNB-website/main/sample-data/training.txt) and [testing](https://raw.githubusercontent.com/lshh125/epiNB-website/main/sample-data/test.txt) data. The first 5 testing data are positive examples, and the last 5 testing data are negative examples.")
    st.markdown("Simply follow the \"Help\" info to get the results.")
    
