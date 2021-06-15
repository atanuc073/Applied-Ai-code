import streamlit as st
from PIL import Image
from clf import is_cancer
import math

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Skin Cancer Detection App")
st.write("This web application is capable of detecting skin cancer. It uses Deep Learning, specifically Convolutional Neural Network to analyze the image.")

st.write("Please upload a photo of your mole or lesion for detection.")

file_up = st.file_uploader("Upload an image", type="jpg")

if file_up is not None:
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Just a second...")
    result, prob = is_cancer(file_up)

    st.write("Result : ", result)

    for key in prob.keys():
        a = prob[key]*100
        a = str(round(a, 2))
        st.write(key, " : ", a, "%")
    # print out the top 5 prediction labels with scores
    # for i in labels:
    #     st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])
