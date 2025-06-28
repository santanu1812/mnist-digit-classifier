import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pickle
from model_class import LogisticRegression
import torch
import cv2
import numpy as np
from PIL import Image,ImageOps

with open('trained_model.pkl','rb') as f:
    model=pickle.load(f)

st.title("BINARY DIGIT CLASSIFIER")
st.write("use the canvas to draw a binary digit")

canvas_result=st_canvas(
    fill_color="rgba(0,0,0,1)",
    stroke_width=20,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=400,
    height=400,
    drawing_mode="freedraw",  
    key="canvas"
)
if canvas_result.image_data is not None:
    st.image(canvas_result.image_data, caption="your doodle", width=150)

img=canvas_result.image_data.astype('uint8')
img = cv2.bitwise_not(img)
img =cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
pil_img=Image.fromarray(img)
bbox=pil_img.getbbox()
if bbox:
    img_cropped=pil_img.crop(bbox)
    img_resized1=img_cropped.resize((20,20),Image.LANCZOS)
    new_img = Image.new('L', (28, 28), color=0)
    upper_left = ((28 - 20) // 2, (28 - 20) // 2)
    new_img.paste(img_resized1, upper_left)
    final_img=new_img
else:
    final_img=Image.new('L',(28,28), color=255)
img_resized2=np.array(final_img)
if img_resized2 is None:
    st.warning("The canvas is empty- enter a digit!")
else:
    img_normalized=img_resized2/255
    img_flatten=img_normalized.flatten().astype(np.float32)
    img_tensor = torch.tensor(img_flatten).unsqueeze(0)
    img_tensor=img_tensor.to('cuda')
    st.image(img_resized2, caption="processed image to be entered into the model", width=150)
if st.button("PREDICT"):
    y_pred=model.predict(img_tensor)
    st.subheader(f" predicted digit = {y_pred}")
