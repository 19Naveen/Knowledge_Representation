from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import Tools
import pandas as pd
import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

llm = None
strict_llm = None

def make_llm(API_KEY):
    """
    Initialize the LLM model with the provided API key.
    
    Args:
        API_KEY (str): The Google Palm API key
    """
    try:
        global llm
        GOOGLE_PALM_API_KEY = API_KEY
        st.session_state.llm = ChatGoogleGenerativeAI(
            google_api_key=GOOGLE_PALM_API_KEY,
            model="gemini-pro",
            temperature=0.5
        )
        global strict_llm
        st.session_state.strict_llm = ChatGoogleGenerativeAI(
            google_api_key=GOOGLE_PALM_API_KEY,
            model="gemini-pro",
            temperature=0.8
        )
    except Exception as e:
        print(f"Error initializing LLM: {e}")

def get_target(sample_data):
    """
    Determine the target variable in the dataset.
    
    Args:
        sample_data (str): Sample records from the dataset
    
    Returns:
        str: The name of the target column
    """
    columns = Tools.fetch_columns()
    
    prompt = '''
        Given a dataset with the following columns:
        {columns}
        
        Here are some sample records from the dataset:
        {sample_data}
        
        Determine which column in this dataset is the target variable and return only the target name.
        return : column_name
        '''  
    query_template = PromptTemplate(template=prompt, input_variables=["columns", "sample_data"])
    query = query_template.format(columns=columns, sample_data=sample_data)
    response = st.session_state.llm.invoke(query)
    return response.content

def dataset_type(csv):
    """
    Determine the type of machine learning problem suitable for the dataset.
    
    Args:
        csv (str): Path to the CSV file
    
    Returns:
        str: The type of machine learning problem (classification, regression, or clustering)
    """
    prompt = '''
        You are a top-tier Machine Learning Engineer who can say what model can be built from the given datasets. 
        Given a dataset, determine whether it is best suited for classification, regression, or clustering. You can only return one of these three options: classification, regression, or clustering.
        Here is the 
        columns:{column}
        dataset:{dataset}'''
    query_template = PromptTemplate(template=prompt, input_variables=["dataset", "column"])
    query = query_template.format(dataset=csv, column=Tools.fetch_columns())
    response = st.session_state.llm.invoke(query)
    return response.content

def generate_insights(csv):
    """
    Generate insights from the dataset.
    
    Args:
        csv (str): Path to the CSV file
    
    Returns:
        str: Generated insights and recommendations
    """
    try:
        columns = Tools.fetch_columns()
        stat = Tools.get_statistical_details()
        prompt = '''
        As a top-tier Data Analyst specializing in extracting insights, your task is to analyze the provided structured dataset to identify key patterns and trends. 
        The dataset at hand ({dataset}) contains the following columns: {Columns} and its stats are {Stats}. Please provide a comprehensive analysis including statistical details and actionable recommendations based on the following insights:
        
        About the Dataset:
        - Explain the dataset and its features in brief
        
        Key Patterns and Trends:
        - Identify significant trends observed in the data.
        - Highlight any correlations between columns.

        Actionable Recommendations:
        - Based on the analysis, propose actionable insights that can enhance decision-making processes.
        '''

        query_template = PromptTemplate(template=prompt, input_variables=["dataset", "Columns", "Stats"])
        query = query_template.format(dataset=csv, Columns=columns, Stats=stat)
        response = st.session_state.llm.invoke(query)
        return response.content

    except Exception as e:
        return f"Error generating insights: {e}"

def generate_and_extract_charts(df):
    """
    Generate and extract chart recommendations based on the dataset.
    
    Args:
        df (pd.DataFrame): The dataset
    
    Returns:
        list: List of dictionaries containing chart specifications
    """
    columns_str = str(list(df.columns))
    stats_str = str(df.describe(include='all').to_dict())
    dtypes_str = str(df.dtypes.apply(lambda x: x.name).to_dict())

    prompt_template = '''
        As a senior Data Analyst specializing in extracting and visualizing insights, your task is to analyze the provided structured dataset and identify key patterns and trends.

        Dataset Columns: {columns}
        Statistical Summary: {stats}
        Data Types: {dtypes}

        Based on your analysis, suggest the most appropriate types of visualizations for these columns. Your goal is to generate at least 5 different graphs from the following options: Line Chart, Pie Chart, Bar Chart, Histogram, Scatter Plot, Box Plot, Heatmap, and Area Chart.

        For each visualization, please:
        1. Specify the columns to be used for the x and y axes (if applicable)
        2. Recommend the chart type
        3. Provide a brief rationale for your choice (1-2 sentences), considering the data types of the columns
        4. Suggest a key insight that might be gleaned from this visualization

        Please return your recommendations in the following format (note that the first index should be the x-axis, the second index should be the y-axis, and the third index should be the chart type):

        Example:
        1. [YearBuilt, Price, Scatter Plot]
           - Rationale: Scatter plots are useful for identifying relationships between two numerical variables.
           - Potential Insight: This visualization could reveal trends such as whether newer properties tend to have higher prices.

        IMPORTANT: Provide at least 5 visualizations.
    '''

    prompt = prompt_template.format(columns=columns_str, stats=stats_str, dtypes=dtypes_str)
    response = st.session_state.llm.invoke(prompt)
    visualizations = Tools.extract_visualization_info(response.content)
    
    return visualizations