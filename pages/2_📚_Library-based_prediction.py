import streamlit as st
import pandas as pd
import epinb

st.title("Library based fully-supervised prediction")
st.markdown("This module uses data from Keskin et al. to make predictions.")
st.markdown("You may choose one or more alleles.")

all_training_data = pd.read_csv("data/Keskin peptide lists filtered.csv")
all_alleles = all_training_data['Allele'].unique().tolist()

col1, col2 = st.columns(2)
with col1:
    selected_alleles = st.multiselect(
        'Choose alleles',
        all_alleles)
        
with col2:
    test_data = st.text_area("Candidate peptides", placeholder='Input peptides here, one per line, or upload below.')
    uploaded_test_data = st.file_uploader("Upload testing peptides")
    
    
col1, col2, col3 = st.columns([1, 3, 5])
runtime_warnings = []

with col1:
    if st.button('Run'):
        training_data = all_training_data[all_training_data['Allele'].isin(selected_alleles)]

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
        
        model = epinb.NBMulti()
        with col3, st.spinner('Training...'):
            model.fit(training_data['Peptide'], training_data['Allele'])
        
        with col3, st.spinner('Running...'):
            res = model.predict_log_odds(test_data, return_df=True, return_best=True)
        
        with col2:
            st.download_button("Download results", res.to_csv(), "epiNB-predictions.csv", disabled=False)
        # st.dataframe(res)
    else:
        with col2:
            st.download_button("Download results", "", disabled=True)
            
if 'box_file_conflict' in runtime_warnings:
    st.warning("When a file is uploaded, the content of the corresponding input box is ignored.\nIf this is not what you want, please remove the uploeded file.")

with st.expander("Step-by-step guide"):
    st.markdown("1. Choose alleles to be used.")
    st.markdown("2. Copy your candidate peptides to be classified into the right text box.")
    st.markdown("3. Click the \"Run\" button.")
    st.markdown("4. Wait till the \"Download results\" button become valid, and click it to download the results.")
