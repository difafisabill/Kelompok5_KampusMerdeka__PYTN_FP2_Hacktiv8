import streamlit as st
import pandas as pd
import requests  # Add this line to import the requests module
import pickle
from PIL import Image
from io import BytesIO
import os
import shutil



url = 'https://content.api.news/v3/images/bin/15b0ed148fe0c2d469b46c1dc4aa8ace'
response = requests.get(url)
if response.status_code == 200:
    image = Image.open(BytesIO(response.content))
    st.image(image, use_column_width=True)
else:
    st.error(f"Failed to download image. Status code: {response.status_code}")

csv_url = 'https://drive.google.com/uc?id=1SD57WXJ8tk8P6_wizagGY8oIBpgQzmUW'



def download_model_from_url(model_url, save_path):
    if model_url.startswith('http'):
        response = requests.get(model_url)
        with open(save_path, 'wb') as file:
            file.write(response.content)
    else:
        shutil.copy(model_url, save_path)

# Logreg
logreg_model_path = 'logreg_model.pkl'
local_logreg_model_path = './logreg.pkl'
download_model_from_url(local_logreg_model_path, logreg_model_path)

if os.path.exists(logreg_model_path):
    with open(logreg_model_path, 'rb') as file:
        logreg_model = pickle.load(file)
else:
    st.error("Failed to load Logistic Regression model.")
# SVM
svm_model_path = 'svm_model.pkl'
local_svm_model_path = './svm.pkl'
download_model_from_url(local_svm_model_path, svm_model_path)

if os.path.exists(svm_model_path):
    with open(svm_model_path, 'rb') as file:
        svm_model = pickle.load(file)
else:
    st.error("Failed to load SVM model.")

st.markdown("<h1 style='text-align: center;'>Apa Besok Akan Hujan ?</h1>",
            unsafe_allow_html=True)
st.markdown("#### Rain Prediction in Australia")
st.markdown("Data ini sudah melalui proses data cleaning dan preprosessing sehingga siap untuk dilakukan pemodelan")


def main():
    st.sidebar.title('Rain or Not')

    @st.cache_resource
    def load_data():
        data = pd.read_csv(csv_url)
        return data

    def get_location_index(loc):
        location_mapping = {
        'Albury': 2,
        'BadgerysCreek': 4,
        'Cobar': 10,
        'CoffsHarbour': 11,
        'Moree': 21,
        'Newcastle': 24,
        'NorahHead': 25,
        'NorfolkIsland': 31,
        'Penrith': 33,
        'Richmond': 34,
        'Sydney': 38,
        'SydneyAirport': 41,
        'WaggaWagga': 43,
        'Williamtown': 9,
        'Wollongong': 36,
        'Canberra': 5,
        'Tuggeranong': 6,
        'MountGinini': 32,
        'Ballarat': 19,
        'Bendigo': 18,
        'Sale': 20,
        'MelbourneAirport': 23,
        'Melbourne': 30,
        'Mildura': 40,
        'Nhil': 12,
        'Portland': 7,
        'Watsonia': 8,
        'Dartmoor': 14,
        'Brisbane': 35,
        'Cairns': 0,
        'GoldCoast': 22,
        'Townsville': 26,
        'Adelaide': 44,
        'MountGambier': 1,
        'Nuriootpa': 42,
        'Woomera': 27,
        'Albany': 29,
        'Witchcliffe': 28,
        'PearceRAAF': 39,
        'PerthAirport': 15,
        'Perth': 17,
        'SalmonGums': 3,
        'Walpole': 13,
        'Hobart': 16,
        'Launceston': 37,
        'AliceSprings': 45,  # Mengubah indeks untuk 'AliceSprings'
        'Darwin': 46,  # Mengubah indeks untuk 'Darwin'
        'Katherine': 47,  # Mengubah indeks untuk 'Katherine'
        'Uluru': 48  # Mengubah indeks untuk 'Uluru'
        }
        return location_mapping.get(loc, -1)
      
    def rain_today(rain):
        rain_mapping={
            'Ya':1,
            'Tidak':0
        }
        return rain_mapping.get(rain,-1)
    
    def rain_tomorrow(predict, location):
        if predict == 0:
            return f'Besok Tidak Hujan di {location}'
        else:
            return f'Besok Hujan di {location}'


        

    data = load_data()
    check_box = st.sidebar.checkbox("Show Dataset")
    if (check_box):
        st.markdown("#### Rain Dataset")
        st.write(data)
    classifier = st.sidebar.selectbox(
        "Klasifikasi", ('Logistic Regression', 'SVM'))
    loc = st.sidebar.selectbox("Lokasi",('Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree',
       'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond',
       'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown',
       'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat',
       'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura',
       'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns',
       'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa',
       'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport',
       'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston',
       'AliceSprings', 'Darwin', 'Katherine', 'Uluru'))
    index = get_location_index(loc)

    TempMin=st.sidebar.slider('Temperatur Minimal', -10.0, 40.0, 1.0)
    Rainfall=st.sidebar.slider('Curah hujan dalam mm', 0.0, 370.0, 1.0)
    Humidity9am=st.sidebar.slider('Humiditas Jam 9 Pagi',  0.0, 100.0, 1.0)
    Humidity3pm=st.sidebar.slider('Humiditas Jam 3 Sore', 0.0, 100.0, 1.0)
    WindGustSpeed=st.sidebar.slider('Keceppatan Angin',  5.0, 135.0, 1.0)
    rain=st.sidebar.radio("Apakah hari ini hujan ?",('Ya','Tidak'))
    index_rain = rain_today(rain)

    def report_display():
        important_feature={
            'Location':index,
            'TempMin':TempMin,
            'Rainfall':Rainfall,
            'Humidity9am': Humidity9am,
            'Humidity3pm' : Humidity3pm,
            'WindGustSpeed':WindGustSpeed,
            'RainToday':index_rain,
        }
        report_data = pd.DataFrame(important_feature, index=[0])
        return report_data
    user_feature = report_display()
    st.markdown('#### Report')
    st.write(user_feature)
    inputs=user_feature
    if st.button('Classify'):
        if classifier == 'Logistic Regression':
            st.success(rain_tomorrow(logreg_model.predict(inputs), loc))
        else:
            st.success(rain_tomorrow(svm_model.predict(inputs), loc))


if __name__ == '__main__':
    main()
