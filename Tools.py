from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import DirectoryLoader
import pandas as pd
import os
import re

import Tools

ORIGINAL_PATH = 'C:\\Users\\Naveen\\Desktop\\DS_Project\\IntelProject\\Data\\CSV\\'
VISUALIZE_PATH = 'C:\\Users\\Naveen\\Desktop\\DS_Project\\IntelProject\\Data\\Visualized_Charts\\'
PATH = 'C:\\Users\\Naveen\\Desktop\\DS_Project\\IntelProject\\Data\\Processed_Data\\'


def load_csv_files(directory_path):
    loader = DirectoryLoader(directory_path, glob="./*.csv", loader_cls=CSVLoader)
    files = loader.load()
    files = [row.page_content for row in files]
    return files


def pd_load_csv_files(directory_path):
    files = os.listdir(directory_path)
    path = os.path.join(directory_path, files[0])
    files = pd.read_csv(path)
    return files


def fetch_columns(csv):
    column_names = re.findall(r'(\w+):', csv)
    return column_names


def get_statistical_details():
    df = pd.read_csv("C:\\Users\\Naveen\\Desktop\\DS_Project\\IntelProject\\Data\\Processed_Data\\Output.csv")
    df = df.describe()
    return df.to_string()


def Visual_rep(response):
    pattern = re.findall("\[['\w, ]+\]", response)
    charts = []
    for i in pattern:
        charts.append(eval(i))
    return charts


def delete_files():
    Path = ''
    for file in os.listdir(PATH):
        os.remove(os.path.join(PATH, file))
    for file in os.listdir(VISUALIZE_PATH):
        os.remove(os.path.join(VISUALIZE_PATH, file))
    for file in os.listdir(ORIGINAL_PATH):
        os.remove(os.path.join(ORIGINAL_PATH, file))


def save_file(uploadedfile, path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)

        with open(os.path.join(path, uploadedfile.name), 'wb') as f:
            f.write(uploadedfile.getbuffer())
    except Exception as e:
        return (f"An error occurred while saving the file: {e}")

    return 1

