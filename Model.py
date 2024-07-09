import pandas as pd
import numpy as np
import streamlit as st
import Tools
import KnowRep
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans

def classification_model_selection(size):
    if size <= 1000:
        return SVC()
    elif size <= 10000:
        return RandomForestClassifier()
    else:
        return XGBClassifier()

def regression_model_selection(size):
    if size <= 1000:
        return LinearRegression()
    elif size <= 10000:
        return RandomForestRegressor()
    else:
        return XGBRegressor()

def clustering_model_selection(size):
    if size <= 1000:
        return KMeans(n_clusters=3)
    elif size <= 10000:
        return AgglomerativeClustering(n_clusters=3)
    else:
        return MiniBatchKMeans(n_clusters=4)

def model_selection(data_type, size):
    if data_type == 'classification':
        return classification_model_selection(size)
    elif data_type == 'regression':
        return regression_model_selection(size)
    else:
        return clustering_model_selection(size)

def build_model(df, data_type, target_column=None):
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    size = len(df)

    if data_type == 'clustering':
        X = df
    else:
        X = df.drop(columns=[target_column])
        Y = df[target_column]

    model = model_selection(data_type, size)

    if data_type != 'clustering':
        model.fit(X, Y)
    else:
        model.fit(X)

    return model

def predict_with_model_input(model, input_data):
    input_data = pd.DataFrame([input_data])
    if hasattr(model, 'predict'):
        return model.predict(input_data)
    elif hasattr(model, 'transform'):
        return model.transform(input_data)
    else:
        raise ValueError("Model does not have a 'predict' or 'transform' method.")

def get_class_names(df, target_column):
    unique_classes = df[target_column].unique()
    return {int(i): str(cls) for i, cls in enumerate(unique_classes)}

def get_cluster_names(predictions):
    unique_clusters = set(int(cluster) for cluster in predictions)
    return {int(cluster): f"Cluster {cluster}" for cluster in unique_clusters}

def get_input_predict(model, data_type, X, class_names=None, cluster_names=None):
    columns = list(X.columns)
    st.write("Columns available for prediction:", columns)

    input_data = {}
    for column in columns:
        if X[column].dtype in ['int64', 'float64']:
            input_data[column] = st.number_input(f"Enter value for {column}:", key=column)
        elif X[column].dtype == 'bool':
            input_data[column] = st.checkbox(f"Enter value for {column}:", key=column)
        else:
            input_data[column] = st.text_input(f"Enter value for {column}:", key=column)

    if st.button('Predict'):
        try:
            predictions = predict_with_model_input(model, input_data)

            if data_type == 'classification' and class_names:
                named_predictions = [class_names.get(int(pred), f"Unknown Class {pred}") for pred in predictions]
            elif data_type == 'clustering' and cluster_names:
                named_predictions = [cluster_names.get(int(pred), f"Unknown Cluster {pred}") for pred in predictions]
            elif data_type == 'regression':
                # For regression, return the raw predicted values
                named_predictions = predictions
            else:
                named_predictions = [float(pred) if isinstance(pred, (np.integer, np.floating)) else pred for pred in predictions]

            return named_predictions
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            return None
    else:
        st.write("Please enter values and click 'Predict'")
        return None

def example_usage(df, data_type, target_column):
    if data_type == 'clustering':
        model = build_model(df, data_type)
        predictions = model.predict(df)
        cluster_names = get_cluster_names(predictions)
        named_predictions = get_input_predict(model, data_type, df, cluster_names=cluster_names)
        st.write("Cluster Names:", {int(k): v for k, v in cluster_names.items()})
    else:
        class_names = {}
        if data_type == 'classification' and target_column:
            class_names = get_class_names(df, target_column)

        model = build_model(df, data_type, target_column)
        X = df.drop(columns=[target_column])
        named_predictions = get_input_predict(model, data_type, X, class_names=class_names)
    
    if named_predictions is not None:
        if data_type == 'regression':
            st.write(f"Predicted value: {named_predictions[0]:.2f}")
        else:
            st.write("Predictions:", named_predictions)

def prediction_model(sample):
    data = Tools.pd_load_csv_files(Tools.PATH)
    csv_type = KnowRep.dataset_type(sample)  
    target = KnowRep.get_target(sample)
    example_usage(data, csv_type, target)
