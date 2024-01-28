import streamlit as st
import pandas as pd
import numpy as np

from transformers import BartTokenizer, BartForConditionalGeneration

sample_text = """
Natural language processing (NLP) is a fascinating field.
In NLP, computers can understand and generate human-like text.
The ultimate goal is to make computers interpret natural language.
This involves challenges such as language ambiguity.
The challenges also include understanding context.
"""

st.title("Speaker Diarization")

st.markdown("---")

st.markdown("## Input")

st.markdown("<p style='font-size: 18px;'>Sample Input Video:</p>",
            unsafe_allow_html=True)

st.video("gitlab_meeting_public.mp4")

st.markdown(
    "<p style='font-size: 18px;'>Please upload a video to perform speaker diariazation:</p>", unsafe_allow_html=True)

uploaded_audio_file = st.file_uploader(
    "", type=["mp4"])

if uploaded_audio_file is None:
    st.write("File uploaded successfully!")

    st.warning("Please be patient, processing might take upto 5 minutes!")

    st.markdown("---")

    st.markdown("## Generated Audio Transcript")
    text_input = st.text_area(
        "", sample_text, height=500)

    model_name = "facebook/bart-base"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    inputs = tokenizer.encode("summarize: " + text_input,
                              return_tensors="pt", max_length=1024, truncation=True)

    summary_ids = model.generate(inputs, max_length=50, min_length=10,
                                 length_penalty=2.0, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    st.markdown("---")

    st.markdown("## Generated Summary")

    st.write(summary)

    st.download_button(
        label="Click here to download the results!",
        data=summary,
        file_name="results.txt",
        key=f"download_button"
    )
