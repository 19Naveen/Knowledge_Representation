import pandas as pd
import streamlit as st
import Tools
import KnowRep


def classification_model_selection(size):
    if size <= 1000:
        from sklearn.svm import SVC
        return SVC()
    elif size <= 10000:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier()
    else:
        from xgboost import XGBClassifier
        return XGBClassifier()


def regression_model_selection(size):
    if size <= 1000:
        from sklearn.linear_model import LinearRegression
        return LinearRegression()
    elif size <= 10000:
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor()
    else:
        from xgboost import XGBRegressor
        return XGBRegressor()


def clustering_model_selection(size):
    if size <= 1000:
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=3)
    elif size <= 10000:
        from sklearn.cluster import AgglomerativeClustering
        return AgglomerativeClustering(n_clusters=3)
    else:
        from sklearn.cluster import MiniBatchKMeans
        return MiniBatchKMeans(n_clusters=4)


def model_selection(data_type, size):
    if data_type == 'classification':
        model = classification_model_selection(size)
    elif data_type == 'regression':
        model = regression_model_selection(size)
    else:
        model = clustering_model_selection(size)
    return model


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
    return df[target_column].unique().tolist()


def get_cluster_names(predictions):
    unique_clusters = set(predictions)
    return {cluster: f"Cluster {cluster}" for cluster in unique_clusters}


def get_input_predict(model, data_type, X, class_names=None, cluster_names=None):
    columns = list(X.columns)
    st.write("Columns available for prediction:", columns)

    input_data = dict()
    for column in columns:
        input_data[column] = st.text_input(f"Enter value for {column}:", key=column)

    # Ensure all inputs are filled before proceeding
    all_inputs_provided = all(input_data[column] for column in columns)

    if st.button('Predict') and all_inputs_provided:
        # Convert input data values to the appropriate types
        input_data = {column: eval(value) for column, value in input_data.items()}
        predictions = predict_with_model_input(model, input_data)

        if data_type == 'classification' and class_names:
            named_predictions = [class_names[pred] for pred in predictions]
        elif data_type == 'clustering' and cluster_names:
            named_predictions = [cluster_names[pred] for pred in predictions]
        else:
            named_predictions = predictions

        return named_predictions
    else:
        st.write("Please enter values for all columns and click 'Predict'")


def example_usage(df, data_type, target_column):
    if data_type == 'clustering':
        model = build_model(df, data_type)
        predictions = model.predict(df)
        cluster_names = get_cluster_names(predictions)
        named_predictions = get_input_predict(model, data_type, df, cluster_names=cluster_names)
        st.write("Cluster Names:", cluster_names)
    else:
        class_names = []
        if data_type == 'classification' and target_column:
            class_names = get_class_names(df, target_column)

        model = build_model(df, data_type, target_column)
        X = df.drop(columns=[target_column])
        named_predictions = get_input_predict(model, data_type, X, class_names=class_names)

    st.write("Predictions:", named_predictions)


def prediction_model(sample):
    data = Tools.pd_load_csv_files(Tools.PATH)
    csv_type = "regression" # KnowRep.dataset_type(sample)
    target = KnowRep.get_target(sample)
    example_usage(data, csv_type, target)
