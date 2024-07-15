import os
import re
import chardet
import pandas as pd

# Constants for file paths
ORIGINAL_PATH = '/workspaces/Knowledge_Representation/Data/CSV/'
VISUALIZE_PATH = '/workspaces/Knowledge_Representation/Data/Visualized_Charts/'
PATH = '/workspaces/Knowledge_Representation/Data/Processed_Data/'

def load_csv_files(directory_path, key='string'):
    """
    Load CSV files from a directory and return either a DataFrame or a string representation.

    Args:
        directory_path (str): Path to the directory containing CSV files.
        key (str): 'dataframe' to return a DataFrame, 'string' for a string representation.

    Returns:
        Loaded data in the specified format.

    Raises:
        ValueError: If no files are found in the directory.
    """
    files = os.listdir(directory_path)
    if not files:
        raise ValueError("No files found in the directory")
    
    path = os.path.join(directory_path, files[0])
    encoding = detect_encoding(path)
    df = pd.read_csv(path, encoding=encoding)
    if key == 'dataframe':
        return df
    else:
        sample_data = df.head().to_dict(orient='records')
        return '\n'.join(f"column{i+1} = {{{', '.join(f'{k!r}: {v}' for k, v in record.items())}}},"
                         for i, record in enumerate(sample_data))


def fetch_columns():
    """
    Fetch column names from the processed CSV file.

    Returns:
        Column names of the processed data.
    """
    df = load_csv_files(PATH, key='dataframe')
    return df.columns


def get_statistical_details():
    """
    Get statistical details of the processed data.

    Returns:
        Statistical summary of the data.
    """
    df = pd.read_csv(os.path.join(PATH, "Output.csv"))
    return df.describe(include='all').to_dict()


def get_dtype():
    """
    Get data types of columns in the processed CSV.

    Returns:
        Column names and their corresponding data types.
    """
    df = pd.read_csv(os.path.join(PATH, "Output.csv"))
    return df.dtypes.apply(lambda x: x.name).to_dict()


import re

def extract_visualization_info(response_content):
    """
    Extract visualization information from a response string.

    Args:
        response_content (str): String containing visualization details.

    Returns:
        List of dictionaries with visualization info.
    """

    insight_pattern = r'Potential Insight: [\w .]+'
    insights = re.findall(insight_pattern, response_content)

    columns_pattern = r'\d+\.\s*\[([^,]+),\s*([^,\]]*)?,?\s*([^\]]+)\]'
    columns = re.findall(columns_pattern, response_content)
 
    result = []
    for i in range(len(columns)):
        column = columns[i]
        info = {
            'x_axis': column[0].strip(),
            'y_axis': column[1].strip() if column[1] and column[1].strip().lower() != 'none' else None,
            'chart_type': column[2].strip(),
            'info': insights[i] if i < len(insights) else None  # Handle case where there are more columns than insights
        }
        result.append(info)
    
    return result


def save_file(uploadedfile, path):
    """
    Save an uploaded file to the specified path.

    Args:
        uploadedfile: File object to be saved.
        path (str): Directory path to save the file.

    Returns:
        1 if successful, error message string if failed.
    """
    try:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, uploadedfile.name), 'wb') as f:
            f.write(uploadedfile.getbuffer())
        return 1
    except Exception as e:
        return f"An error occurred while saving the file: {e}"


def make_folders():
    """Create necessary directories if they don't exist."""
    for directory in [ORIGINAL_PATH, PATH, VISUALIZE_PATH]:
        os.makedirs(directory, exist_ok=True)


def delete_files():
    """Delete all files in the specified directories."""
    for directory in [PATH, VISUALIZE_PATH, ORIGINAL_PATH]:
        for file in os.listdir(directory):
            os.remove(os.path.join(directory, file))


def detect_encoding(file_path):
    """
    Detect the encoding of a given file.

    Args:
        file_path (str): Path to the file.

    Returns:
        Detected encoding of the file.
    """
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    return chardet.detect(raw_data)['encoding']