# import package
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# membaca dataset
data = pd.read_csv("Data Clean.csv")
st.title("Aplikasi Prediksi Harga Rumah")

# cek data
st.write("Aplikasi ini dibuat untuk memperkirakan rentang harga rumah yang akan dibeli oleh konsumen")
check_data = st.checkbox("Lihat data yang ada")
if check_data:
    st.write(data)
st.subheader("Ayo mulai prediksi harga rumah impianmu!")

# input angka
sqft_liv = st.slider("Berapa ukuran ruang tamu yang kamu inginkan (ft2)?", int(data.sqft_living.min()), int(data.sqft_living.max()), int(data.sqft_living.mean()))
bath = st.slider("Berapa banyak kamar mandi?", int(data.bathrooms.min()), int(data.bathrooms.max()), int(data.bathrooms.mean()))
bed = st.slider("Berapa banyak tempat tidur?", int(data.bedrooms.min()), int(data.bedrooms.max()), int(data.bedrooms.mean()))
floor = st.slider("Berapa lantai rumah yang kamu inginkan?", int(data.floors.min()), int(data.floors.max()), int(data.floors.mean()))

# membagi data
X = data.drop('price', axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=45)

# import model
model = LinearRegression()
# memodelkan dan melakukan prediksi
model.fit(X_train, y_train)
model.predict(X_test)
errors = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
predictions = model.predict([[sqft_liv, bath, bed, floor]])[0]

#cek prediksi harga rumah
if st.button("Jalankan!"):
    st.header("Prediksi harga rumah impianmu = {} $".format(int(predictions)))
    st.subheader("Range harga rumahmu = {} $ - {} $".format(int(predictions-errors),int(predictions+errors)))
