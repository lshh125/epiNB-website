import streamlit as st
import epinb

st.markdown("# Mono-allelic prediction")

st.markdown("This module is for predict peptides presented to a single HLA allele.")

col1, col2 = st.columns(2)
with col1:
    training_data = st.text_area("Training peptides", placeholder='Input peptides here, one per line, or upload below.')
    uploaded_training_data = st.file_uploader("Upload training peptides")
with col2:
    test_data = st.text_area("Candidate peptides", placeholder='Input peptides here, one per line, or upload below.')
    uploaded_test_data = st.file_uploader("Upload testing peptides")
     
col1, col2, col3 = st.columns([1, 3, 5])

runtime_warnings = []

with col1:
    if st.button('Run'):
        if uploaded_training_data is None:
            training_data = training_data.strip().split()
        else:
            if training_data != "":
                runtime_warnings.append('box_file_conflict')
            training_data = uploaded_training_data.getvalue().decode('UTF-8').strip().split()
            
        if uploaded_test_data is None:
            test_data = test_data.strip().split()
        else:
            if test_data != "":
                runtime_warnings.append('box_file_conflict')
            test_data = uploaded_test_data.getvalue().decode('UTF-8').strip().split()
        
        if len(training_data) < 10:
            with col0:
                st.error("At least 10 training peptides needed.")
            
        if len(test_data) < 1:
            with col0:
                st.error("At least 1 andidate peptides required.")
        
        model = epinb.NBScore()
        with col3, st.spinner('Training...'):
            model.fit(training_data)
        
        with col3, st.spinner('Running...'):
            res = model.predict_details(test_data)
        
        with col2:
            st.download_button("Download results", res.to_csv(), "epiNB-predictions.csv", disabled=False)
        # st.dataframe(res)
    else:
        with col2:
            st.download_button("Download results", "", disabled=True)

if 'box_file_conflict' in runtime_warnings:
    st.warning("When a file is uploaded, the content of the corresponding input box is ignored.\nIf this is not what you want, please remove the uploeded file.")

with st.expander("Step-by-step guide"):
    st.markdown("1. Copy your training peptides into the left text box.")
    st.markdown("2. Copy your candidate peptides to be classified into the right text box.")
    st.markdown("3. Click the \"Run\" button.")
    st.markdown("4. Wait till the \"Download results\" button become valid, and click it to download the results.")
    
with st.expander("Want some sample input to try out?"):
    st.markdown("We used data for A0203 as an example here.")
    st.markdown("You can download sample [training](https://raw.githubusercontent.com/lshh125/epiNB-website/main/sample-data/training.txt) and [testing](https://raw.githubusercontent.com/lshh125/epiNB-website/main/sample-data/test.txt) data. The first 5 testing data are positive examples, and the last 5 testing data are negative examples.")
    st.markdown("Simply follow the \"Help\" info to get the results.")
    
