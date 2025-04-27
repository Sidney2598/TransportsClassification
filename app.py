from fastai.vision.all import *
import streamlit as st
import pathlib
import plotly.express as px
import platform
from fastai.learner import load_learner

plt=platform.system()
if plt=='Linux':pathlib.PosixPath=pathlib.WindowsPath

st.title('Nihoyat ishladi')
file=st.file_uploader('Rasm yuklash',type=['png','jpeg','gif','svg'])
if file:
    st.image(file)
    #PIL CONVERT
    img=PILImage.create(file)
    #model
    model=load_learner('transport_1_loyiha.pkl',cpu=True)

    predict, pred_id, probs=model.predict(img)
    st.success(f'Bashorat:{predict}')
    st.info(f'Ehtimollik:{probs[pred_id]*100:.1f}%')    
    #plotly
    fig=px.bar(x=probs,y=model.dls.vocab)
    st.plotly_chart(fig)