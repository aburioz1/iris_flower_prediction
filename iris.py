import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import sklearn
import pickle
iris_model = pickle.load(open("C:\\Users\\HP\\Downloads\\iris_model.pkl", "rb"))
st.title("Iris Flower Prediction App")
image = Image.open('C:\\Users\\HP\\Downloads\\pexels-elina-nova-13797008.jpg')
st.image(image, width= 300)
def user_report():
    sepal_length = st.sidebar.slider('sepal length',1.0,10.0,0.1)
    sepal_width = st.sidebar.slider('sepal width',1.0,10.0,0.1)
    petal_length = st.sidebar.slider('petal length',1.0,10.0,0.1)
    petal_width = st.sidebar.slider('petal width',1.0,10.0,0.1)
    data_report = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    data = pd.DataFrame(data_report, index=[0])
    return data
user_data = user_report()
st.write(user_data)
prediction = iris_model.predict(user_data)
st.success(prediction)
if (prediction ==0):
    st.success('setosa')
    image = Image.open('C:\\Users\\HP\\Downloads\\Irissetosa1.jpg')
    st.image(image, width=300)
elif (prediction ==1):
    st.success('versicolor')
    image = Image.open('C:\\Users\\HP\\Downloads\\Blue_Flag,_Ottawa.jpg')
    st.image(image, width=300)
else:
    st.success('virginica')
    image = Image.open('C:\\Users\\HP\\Downloads\\Iris_virginica_2.jpg')
    st.image(image, width=300)


