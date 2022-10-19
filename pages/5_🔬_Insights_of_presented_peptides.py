import streamlit as st
import epinb

import logomaker
import wordcloud
import matplotlib.pyplot as plt
from scipy.stats import entropy

st.title("Insights of presented peptides")


training_data = st.text_area("Training peptides", placeholder='Input peptides here, one per line, or upload below.')
uploaded_training_data = st.file_uploader("Upload training peptides")

     
col1, col2, col3 = st.columns([1, 3, 5])

runtime_warnings = []

if st.button('Run'):
    if uploaded_training_data is None:
        training_data = training_data.strip().split()
    else:
        if training_data != "":
            runtime_warnings.append('box_file_conflict')
        training_data = uploaded_training_data.getvalue().decode('UTF-8').strip().split()
    
    if len(training_data) < 10:
        st.error("At least 10 training peptides needed.")
        
    m = epinb.NBScore()
    with col3, st.spinner('Training...'):
        m.fit(training_data)

    temp = m.fit_details_1().T.reset_index(drop=True)
    temp = temp * (3 - entropy(temp.T).reshape([-1, 1]))

    fig, ax = plt.subplots(1,1,figsize=[4.5,1.25])
    ww_logo = logomaker.Logo(temp,
                         color_scheme='NajafabadiEtAl2017',
                        ax=ax,
                         vpad=.1,
                         width=.8)
    ww_logo.ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 7, 8, 9, 0])
    ww_logo.ax.set_ylabel('information (nits)')
        
    with st.expander("1st-order motifs", expanded=True):
        fig
    with st.expander("2nd-order pan-allelic motifs", expanded=True):
        fit_details_2 = m.fit_details_2()
        tabs = st.tabs(fit_details_2.columns[:10].tolist())
        for i, dac in enumerate(fit_details_2.columns[:10]):
            with tabs[i]:
                fig, ax = plt.subplots()
                wc = wordcloud.WordCloud(width = 600, height = 250, min_font_size = 10, background_color ='white', collocations=False)
                wc.fit_words(fit_details_2[dac].to_dict())
                ax.imshow(wc)
                ax.axis("off")
                st.pyplot(fig)
                
    with st.expander("2nd-order allele-specific motifs", expanded=True):
        fit_details_2 = m.fit_details_2()
        tabs = st.tabs(fit_details_2.columns[10:].tolist())
        for i, dac in enumerate(fit_details_2.columns[10:]):
            with tabs[i]:
                fig, ax = plt.subplots()
                wc = wordcloud.WordCloud(width = 600, height = 250, min_font_size = 10, background_color ='white', collocations=False)
                wc.fit_words(fit_details_2[dac].to_dict())
                ax.imshow(wc)
                ax.axis("off")
                st.pyplot(fig)
                
if 'box_file_conflict' in runtime_warnings:
    st.warning("When a file is uploaded, the content of the corresponding input box is ignored.\nIf this is not what you want, please remove the uploeded file.")



with st.expander("Step-by-step guide"):
    st.markdown("1. Copy your training peptides into the left text box.")
    st.markdown("2. Click the \"Run\" button. The motifs will show below momentarily.")
    
with st.expander("Want some sample input to try out?"):
    st.markdown("We used data for A0203 as an example here.")
    st.markdown("You can download sample [training](https://raw.githubusercontent.com/lshh125/epiNB-website/main/sample-data/training.txt).")
    st.markdown("Simply follow the \"Help\" info to get the results.")
    
