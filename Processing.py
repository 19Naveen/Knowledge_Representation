import Tools
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer


def preprocess_dataset():
    try:
        df = Tools.load_csv_files(Tools.ORIGINAL_PATH, key='dataframe')
    except Exception as e:
        raise ValueError(f"Error loading dataset: {e}")

    if df.empty:
        raise ValueError("The dataset is empty.")

    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ['int64', 'float64']:
                imputer = SimpleImputer(strategy='mean')
                df[col] = imputer.fit_transform(df[[col]]).flatten()
            elif df[col].dtype == 'object':
                imputer = SimpleImputer(strategy='most_frequent')
                df[col] = imputer.fit_transform(df[[col]]).flatten()
            else:
                raise ValueError(f"Column {col} has unsupported data type {df[col].dtype} for imputation.")

    def remove_outliers(df, col):
        if df[col].dtype in ['int64', 'float64']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df

    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df = remove_outliers(df, col)

    #df = df.sample().reset_index(drop=True)
    df.to_csv(f"{Tools.PATH}/Output.csv", index=False)


def Visualize_charts(charts):
    df = Tools.pd_load_csv_files(Tools.ORIGINAL_PATH)
    df = df.loc[:25]
    save_path = '/workspaces/Knowledge_Representation/Data/Visualized_Charts'
    
    for chart in charts:
        x_col, y_col, chart_type = chart[0], chart[1], chart[2]

        if df[x_col].dtype == 'object' or (y_col and df[y_col].dtype == 'object'):
            continue

        if chart_type == 'Scatter Plot' and y_col:
            plt.scatter(df[x_col], df[y_col])
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f"Scatter Plot: {x_col} vs {y_col}")
            plt.savefig(f"{save_path}/scatter_plot_{x_col}_{y_col}.png")

        elif chart_type == 'Bar Chart':
            labels = df[x_col].value_counts().index
            sizes = df[x_col].value_counts().values
            plt.bar(labels, sizes)
            plt.xlabel(x_col)
            plt.ylabel('Counts')
            plt.title(f"Bar Chart: {x_col}")
            plt.savefig(f"{save_path}/bar_chart_{x_col}.png")

        elif chart_type == 'Histogram':
            plt.hist(df[x_col], bins=30, color='blue', alpha=0.7)
            plt.xlabel(x_col)
            plt.ylabel('Frequency')
            plt.title(f"Histogram: {x_col}")
            plt.savefig(f"{save_path}/histogram_{x_col}.png")

        elif chart_type == 'Pie Chart':
            labels = df[x_col].value_counts().index
            sizes = df[x_col].value_counts().values
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
            plt.axis('equal')
            plt.title(f"Pie Chart: {x_col}")
            plt.savefig(f"{save_path}/pie_chart_{x_col}.png")

        elif chart_type == 'Box Plot':
            sns.boxplot(x=df[x_col])
            plt.xlabel(x_col)
            plt.title(f"Box Plot: {x_col}")
            plt.savefig(f"{save_path}/box_plot_{x_col}.png")

        elif chart_type == 'Heatmap' and y_col:
            data = df.pivot_table(index=x_col, columns=y_col, aggfunc='size', fill_value=0)
            sns.heatmap(data, annot=True, fmt="d")
            plt.title(f"Heatmap: {x_col} vs {y_col}")
            plt.savefig(f"{save_path}/heatmap_{x_col}_{y_col}.png")

        elif chart_type == 'Area Chart' and y_col:
            df.plot.area(x=x_col, y=y_col, alpha=0.4)
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f"Area Chart: {x_col} vs {y_col}")
            plt.savefig(f"{save_path}/area_chart_{x_col}_{y_col}.png")

        elif chart_type == 'Line Chart' and y_col:
            plt.plot(df[x_col], df[y_col])
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f"Line Chart: {x_col} vs {y_col}")
            plt.savefig(f"{save_path}/line_chart_{x_col}_{y_col}.png")

        else:
            print(f"Unknown or unsupported chart type: {chart_type}. Skipping {x_col} vs {y_col}.")
