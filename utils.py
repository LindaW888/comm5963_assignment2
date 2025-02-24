import requests
import os
import sklearn
import pandas as pd
from static import PREDICTOR_COLUMNS, TARGET_COLUMN

def read_dataframe() -> pd.DataFrame:
    file_name = 'Iris.xls'
    if os.path.exists(file_name):
        print(f'Reusing previously downloaded file: {file_name}')
    else:
        # Data Reference: https://doi.org/10.7910/DVN/R2RGXR
        # Columns: ['Species_No', 'Petal_width', 'Petal_length', 'Sepal_width', 'Sepal_length', 'Species_name']
        url_iris = 'http://faculty.smu.edu/tfomby/eco5385_eco6380/data/Iris.xls'
        print(f'File did not exists, downloading: {url_iris}')
        with open(file_name, 'wb') as file_obj:
            file_obj.write(requests.get(url_iris).content)
    df = pd.read_excel(file_name)
    # Adjust the species_no to start from 0
    df['Species_No'] = df['Species_No'] - 1
    return df

def load_train_test_datasets():
    df_iris = read_dataframe()
    # Train-Test split
    train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(
        df_iris[PREDICTOR_COLUMNS], df_iris[TARGET_COLUMN], test_size=0.2, random_state=5963)
    return train_x, train_y, test_x, test_y