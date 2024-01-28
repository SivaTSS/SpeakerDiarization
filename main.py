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

uploaded_audio_file = st.file_uploader(
    "**Upload an audio file**", type=["mp3"])

if uploaded_audio_file is not None:
    st.write("File uploaded successfully!")

generate_audio_transcript = st.button("Generate Audio Transcript")

st.markdown("---")

st.markdown("## Audio Transcript")
text_input = st.text_area(
    "", sample_text, height=500)

model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

inputs = tokenizer.encode("summarize: " + text_input,
                          return_tensors="pt", max_length=1024, truncation=True)

summary_ids = model.generate(inputs, max_length=50, min_length=10,
                             length_penalty=2.0, num_beams=4, early_stopping=True)

summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

st.markdown("---")

st.markdown("## Meeting Summary")

st.write(summary)
