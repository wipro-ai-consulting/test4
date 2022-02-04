from transformers import pipeline

import streamlit as st

from transformers import AutoTokenizer
from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer

 
st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Wipro_Primary_Logo_Color_RGB.svg/240px-Wipro_Primary_Logo_Color_RGB.svg.png')
st.title('Medical Affairs Use case: Can AI understand special words used by doctors? ')
st.write('Can the AI spot elements like CITY  names in any sentance? Try using the 1st model in the dropdown list in the leftside') 
st.write('Can the AI spot CHEMICAL Names, Diseases Names in any sentance?  Try using the 3rd model ') 

st.header('To try out, Enter any text below and presss Control + Enter on keyboard') 
st.header('Also try out different AI brains. One of models can spot Diseases names! Select a brain on the drop down on left side of this screen.') 

  


 
tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-pubmed")

model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-pubmed", attention_type="original_full") 

text = "Amoxicillin is a broad-spectrum antibacterial that has been available for clinical use in a wide range of indications for over 20 years and is now used"
inputs = tokenizer(text, return_tensors='pt')
prediction = model.generate(**inputs)
predictionSENTANCE = tokenizer.batch_decode(prediction)

st.write(predictionSENTANCE)
