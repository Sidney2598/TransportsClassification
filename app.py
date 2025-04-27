from fastai.vision.all import *
import streamlit as st
import pathlib
import plotly.express as px
import platform

# Linux server uchun WindowsPath to PosixPath
if platform.system() == 'Linux':
    pathlib.WindowsPath = pathlib.PosixPath

# Modelni dastur boshlanishida yuklab olamiz
model = load_learner('transport_1_loyiha.pkl', cpu=True)

st.title('Nihoyat ishladi')
file = st.file_uploader('Rasm yuklash', type=['png', 'jpeg', 'gif', 'svg'])

if file:
    st.image(file)

    # Rasmni PIL formatga o'zgartirish
    img = PILImage.create(file)

    # Modeldan bashorat olish
    predict, pred_id, probs = model.predict(img)

    st.success(f'Bashorat: {predict}')
    st.info(f'Ehtimollik: {probs[pred_id]*100:.1f}%')

    # Ehtimolliklarni chizish
    fig = px.bar(x=probs, y=model.dls.vocab)
    st.plotly_chart(fig)
