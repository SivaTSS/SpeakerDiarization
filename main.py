# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np

from summarizer import Summarizer

st.title("AI tool for meetings summarization")

st.markdown("---")

sample_text = """
Natural language processing (NLP) is a field of artificial intelligence that focuses on the interaction 
between computers and humans through natural language. The ultimate objective of NLP is to enable computers 
to understand, interpret, and generate human-like text. It involves several challenges, such as language 
ambiguity, context understanding, and the nuances of human communication.

There are two main approaches to text summarization: extractive and abstractive. Extractive summarization 
involves selecting and presenting the most important sentences or phrases from the original text, while 
abstractive summarization aims to generate new sentences that convey the main ideas in a more concise form.

The 'summarizer' library uses BERT for extractive summarization. Let's generate a summary for the sample text.
"""

text_input = st.text_area(
    "**Enter/Paste text for summarization:**", sample_text, height=500)

generate_summary = st.button("Generate Summary")

if generate_summary:

    bert_summarizer = Summarizer()

    st.markdown(bert_summarizer(text_input))
