import src.Tools as Tools
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import numpy as np


def preprocess_dataset():
    """
    Preprocess the dataset by handling missing values and removing outliers.
   
    Raises:
        ValueError: If there's an error loading the dataset or if it's empty.
    """
    try:
        df = Tools.load_csv_files(Tools.ORIGINAL_PATH, key='dataframe')
    except Exception as e:
        raise ValueError(f"Error loading dataset: {e}")
    
    if df.empty:
        raise ValueError("The dataset is empty.")
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The loaded data is not a pandas DataFrame.")
    df.dropna(axis=1, how='all', inplace=True)
    
    if df.empty:
        raise ValueError("All columns were empty and have been removed.")
    
    for col in df.columns:
        if df[col].isnull().all():
            print(f"Warning: Column '{col}' is entirely empty. Dropping this column.")
            df.drop(col, axis=1, inplace=True)
            continue
        
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
            elif df[col].dtype == 'object':
                imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
            else:
                print(f"Warning: Column {col} has unsupported data type {df[col].dtype} for imputation. Skipping.")
                continue
            
            df[col] = imputer.fit_transform(df[[col]]).flatten()
    
    if df.empty:
        raise ValueError("All columns have been removed due to being empty or having unsupported data types.")
    df = remove_outliers(df)
    output_path = os.path.join(Tools.PATH, "Output.csv")
    try:
        df.to_csv(output_path, index=False)
    except Exception as e:
        raise IOError(f"Error saving processed dataset: {e}")


def remove_outliers(df):
    """
    Remove outliers from numerical columns using the IQR method.
   
    Args:
        df (pd.DataFrame): Input DataFrame
   
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_columns:
        if df[col].nunique() > 1:  # Only process columns with more than one unique value
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df


def Visualize_charts(charts):
    """
    Create and save various types of charts based on the input specifications using Seaborn.

    Args:
        charts (list): List of dictionaries containing chart specifications

    Returns:
        None
    """
    df = Tools.load_csv_files(Tools.ORIGINAL_PATH, key='dataframe')
    save_path = './Data/Visualized_Charts'
    os.makedirs(save_path, exist_ok=True)
    
    sns.set_theme(style="whitegrid", palette="deep", font="sans-serif")
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    
    for chart in charts:
        x_col, y_col, chart_type = chart['x_axis'], chart['y_axis'], chart['chart_type']

        if x_col not in df.columns or (y_col and y_col not in df.columns):
            print(f"Column(s) not found in the dataframe. Skipping.")
            continue


        try:
            sample_size = min(1000, len(df))
            if chart_type == 'Scatter Plot':
                df_sample = df.sample(n=50, random_state=42)
            else:
                df_sample = df.sample(n=sample_size, random_state=42)

            if chart_type == 'Scatter Plot' and y_col:
                plt.figure(figsize=(12, 8))
                sns.scatterplot(data=df_sample, x=x_col, y=y_col, alpha=0.6)
                plt.title(f"Scatter Plot: {x_col} vs {y_col}")
            elif chart_type == 'Bar Chart':
                plt.figure(figsize=(12, 8))
                if df[x_col].dtype == 'object':
                    data = df[x_col].value_counts().nlargest(20)
                    sns.barplot(x=data.index, y=data.values)
                    plt.ylabel('Count')
                elif y_col:
                    data = df.groupby(x_col)[y_col].mean().nlargest(20)
                    sns.barplot(x=data.index, y=data.values)
                    plt.ylabel(f'Mean {y_col}')
                else:
                    sns.histplot(df[x_col], kde=True)
                    plt.ylabel('Frequency')
                plt.xlabel(x_col)
                plt.title(f"Bar Chart: {x_col}")
                plt.xticks(rotation=45, ha='right')
            elif chart_type == 'Histogram':
                plt.figure(figsize=(12, 8))
                sns.histplot(df_sample[x_col], kde=True)
                plt.xlabel(x_col)
                plt.ylabel('Frequency')
                plt.title(f"Histogram: {x_col}")
            elif chart_type == 'Pie Chart':
                plt.figure(figsize=(12, 8))
                if df[x_col].dtype == 'object':
                    data = df[x_col].value_counts().nlargest(10)
                    plt.pie(data.values, labels=data.index, autopct='%1.1f%%', startangle=90)
                    plt.title(f"Pie Chart: Top 10 categories of {x_col}")
                else:
                    print(f"Pie Chart not suitable for numerical data: {x_col}. Skipping.")
            elif chart_type == 'Box Plot':
                plt.figure(figsize=(12, 8))
                sns.boxplot(x=df_sample[x_col])
                plt.xlabel(x_col)
                plt.title(f"Box Plot: {x_col}")
            elif chart_type == 'Heatmap' and y_col:
                if df[x_col].dtype == 'object' and df[y_col].dtype == 'object':
                    pivot_table = pd.crosstab(df[x_col], df[y_col])
                    if pivot_table.empty:
                        print("Pivot table is empty. Skipping heatmap.")
                        continue
                
                    sns.heatmap(pivot_table, annot=True, fmt="d", cmap="YlGnBu")
                    plt.title(f"Heatmap: {x_col} vs {y_col}")
                    plt.xlabel(x_col)
                    plt.ylabel(y_col)
                elif pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
                    correlation = df[[x_col, y_col]].corr()
                    sns.heatmap(correlation, annot=True, cmap="YlGnBu")
                    plt.title(f"Correlation Heatmap: {x_col} vs {y_col}")
                else:
                    print(f"Heatmap requires two categorical variables or two numeric variables. Skipping {x_col} vs {y_col}.")
                    continue
            elif chart_type in ['Area Chart', 'Line Chart'] and y_col:
                plt.figure(figsize=(12, 8))
                if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
                    df_sorted = df_sample.sort_values(by=x_col)
                    if chart_type == 'Area Chart':
                        sns.stackplot(data=df_sorted, x=x_col, y=y_col)
                        plt.fill_between(df_sorted[x_col], df_sorted[y_col], alpha=0.3)
                    else:  # Line Chart
                        sns.lineplot(data=df_sorted, x=x_col, y=y_col)
                    plt.xlabel(x_col)
                    plt.ylabel(y_col)
                    plt.title(f"{chart_type}: {x_col} vs {y_col}")
                else:
                    print(f"{chart_type} requires numerical data for both axes. Skipping {x_col} vs {y_col}.")
            else:
                print(f"Unknown or unsupported chart type: {chart_type}. Skipping.")
                continue
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f"{chart_type.lower().replace(' ', '_')}_{x_col}_{y_col if y_col else ''}.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating chart for {chart_type} with {x_col} and {y_col}: {str(e)}")

    print(f"Charts have been saved to {save_path}")