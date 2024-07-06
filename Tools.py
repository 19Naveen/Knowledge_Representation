from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import DirectoryLoader
import pandas as pd
import os
import re
import chardet

# 'Get Your API KEY From https://aistudio.google.com/app/apikey '

ORIGINAL_PATH = '/workspaces/Knowledge_Representation/Data/CSV/'
VISUALIZE_PATH = '/workspaces/Knowledge_Representation/Data/Visualized_Charts/'
PATH = '/workspaces/Knowledge_Representation/Data/Processed_Data/'


def load_csv_files(directory_path, key='string'):
    files = os.listdir(directory_path)
    if not files:
        raise ValueError("No files found in the directory")
    path = os.path.join(directory_path, files[0])
    encoding = detect_encoding(path)
    df = pd.read_csv(path, encoding=encoding)
    #df = df.sample(n = (df.size/2)).reset_index(drop=True)

    if key == 'dataframe':
        return df
    else:
        sample_data = df.head().to_dict(orient='records')
        output_str = ""
        for i, record in enumerate(sample_data):
            column_name = f"column{i+1}"
            record_str = ", ".join([f"'{k}': {v}" for k, v in record.items()])
            output_str += f"{column_name} = {{{record_str}}},\n"
        return output_str
    

def fetch_columns():
    df = load_csv_files(PATH, key='dataframe')
    return df.columns


def get_statistical_details():
    df = pd.read_csv("/workspaces/Knowledge_Representation/Data/Processed_Data/Output.csv")
    df = df.describe()
    return df.to_string()


def Visual_rep(response):
    pattern = re.findall("\[['\w, ]+\]", response)
    charts = []
    for i in pattern:
        charts.append(eval(i))
    return charts


def save_file(uploadedfile, path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)

        with open(os.path.join(path, uploadedfile.name), 'wb') as f:
            f.write(uploadedfile.getbuffer())
    except Exception as e:
        return (f"An error occurred while saving the file: {e}")

    return 1


def make_folders():
    if not os.path.exists(ORIGINAL_PATH):
        os.makedirs(ORIGINAL_PATH)
    
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    if not os.path.exists(VISUALIZE_PATH):
        os.makedirs(VISUALIZE_PATH)


def delete_files():
    for file in os.listdir(PATH):
        os.remove(os.path.join(PATH, file))
    for file in os.listdir(VISUALIZE_PATH):
        os.remove(os.path.join(VISUALIZE_PATH, file))
    for file in os.listdir(ORIGINAL_PATH):
        os.remove(os.path.join(ORIGINAL_PATH, file))


def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    return result['encoding']