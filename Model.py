import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, silhouette_score

def prepare_for_model(df, target_variable, pca_threshold=0.95, variance_threshold=0.01):
    """
    Prepare the dataframe for classification or regression model training.
    
    Args:
    df (pd.DataFrame): Input dataframe.
    target_variable (str): Name of the target variable column.
    pca_threshold (float): Explained variance ratio threshold for PCA.
    variance_threshold (float): Threshold for variance-based feature selection.
    
    Returns:
    X (np.array): Prepared feature matrix.
    y (np.array): Target variable.
    feature_names (list): List of feature names after preprocessing.
    """
    X = df.drop(columns=[target_variable])
    y = df[target_variable].values
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_cats = encoder.fit_transform(X[categorical_features])
    encoded_feature_names = encoder.get_feature_names_out(categorical_features)
    
    scaler = StandardScaler()
    scaled_numeric = scaler.fit_transform(X[numeric_features])
    
    X_preprocessed = np.hstack((scaled_numeric, encoded_cats))
    feature_names = list(numeric_features) + list(encoded_feature_names)
    
    selector = VarianceThreshold(threshold=variance_threshold)
    X_selected = selector.fit_transform(X_preprocessed)
    selected_mask = selector.get_support()
    feature_names = [f for f, selected in zip(feature_names, selected_mask) if selected]
    
    if X_selected.shape[1] > 100:
        pca = PCA(n_components=pca_threshold, svd_solver='full')
        X_pca = pca.fit_transform(X_selected)
        feature_names = [f"PC_{i+1}" for i in range(X_pca.shape[1])]
        return X_pca, y, feature_names
    else:
        return X_selected, y, feature_names

def prepare_for_clustering(df, pca_threshold=0.95, variance_threshold=0.01):
    """
    Prepare the dataframe for clustering model training.
    
    Args:
    df (pd.DataFrame): Input dataframe.
    pca_threshold (float): Explained variance ratio threshold for PCA.
    variance_threshold (float): Threshold for variance-based feature selection.
    
    Returns:
    X (np.array): Prepared feature matrix.
    feature_names (list): List of feature names after preprocessing.
    """
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object', 'category']).columns
    
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_cats = encoder.fit_transform(df[categorical_features])
    encoded_feature_names = encoder.get_feature_names_out(categorical_features)
    
    scaler = StandardScaler()
    scaled_numeric = scaler.fit_transform(df[numeric_features])
    
    X_preprocessed = np.hstack((scaled_numeric, encoded_cats))
    feature_names = list(numeric_features) + list(encoded_feature_names)
    
    selector = VarianceThreshold(threshold=variance_threshold)
    X_selected = selector.fit_transform(X_preprocessed)
    selected_mask = selector.get_support()
    feature_names = [f for f, selected in zip(feature_names, selected_mask) if selected]
    
    if X_selected.shape[1] > 100:
        pca = PCA(n_components=pca_threshold, svd_solver='full')
        X_pca = pca.fit_transform(X_selected)
        feature_names = [f"PC_{i+1}" for i in range(X_pca.shape[1])]
        return X_pca, feature_names
    else:
        return X_selected, feature_names

def classification_model_selection(size):
    """
    Select a classification model based on dataset size.
    
    Args:
    size (int): Number of samples in the dataset.
    
    Returns:
    model: Selected classification model.
    """
    if size <= 1000:
        return SVC()
    elif size <= 10000:
        return RandomForestClassifier()
    else:
        return XGBClassifier()

def regression_model_selection(size):
    """
    Select a regression model based on dataset size.
    
    Args:
    size (int): Number of samples in the dataset.
    
    Returns:
    model: Selected regression model.
    """
    if size <= 1000:
        return LinearRegression()
    elif size <= 10000:
        return RandomForestRegressor()
    else:
        return XGBRegressor()

def clustering_model_selection(size):
    """
    Select a clustering model based on dataset size.
    
    Args:
    size (int): Number of samples in the dataset.
    
    Returns:
    model: Selected clustering model.
    """
    if size <= 1000:
        return KMeans(n_clusters=3)
    elif size <= 10000:
        return AgglomerativeClustering(n_clusters=3)
    else:
        return MiniBatchKMeans(n_clusters=4)

def model_selection(data_type, size):
    """
    Select an appropriate model based on data type and size.
    
    Args:
    data_type (str): Type of the task - 'classification', 'regression', or 'clustering'.
    size (int): Number of samples in the dataset.
    
    Returns:
    model: Selected machine learning model.
    """
    if data_type == 'classification':
        return classification_model_selection(size)
    elif data_type == 'regression':
        return regression_model_selection(size)
    else:
        return clustering_model_selection(size)

def prediction_model(df, target_variable, data_type, user_input):
    """
    Main function to predict model based on the input data.
    
    Args:
    sample_file (str): Path to the input CSV file.
    """

    user_input = user_input.split(',')
    if data_type == 'clustering':
        X_selected, feature_names = prepare_for_clustering(df)
    else:
        X_selected, y, feature_names = prepare_for_model(df, target_variable)

    size = len(df)
    model = model_selection(data_type, size)

    preprocessor = {}

    if data_type != 'clustering':
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.1, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if data_type == 'classification':
            print('Accuracy:', accuracy_score(y_test, y_pred))
            print(classification_report(y_test, y_pred))
        elif data_type == 'regression':
            print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
            print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
            print('R-squared:', model.score(X_test, y_test))
    else:
        model.fit(X_selected)
        labels = model.labels_
        print('Silhouette Score:', silhouette_score(X_selected, labels))

    preprocessor['scaler'] = StandardScaler().fit(X_selected)
    if X_selected.shape[1] > 100:
        preprocessor['pca'] = PCA(n_components=0.95, svd_solver='full').fit(X_selected)


    print('Enter values for the following features, separated by commas:')
    print(', '.join(df.columns.drop(target_variable) if data_type != 'clustering' else df.columns))
    user_input = input().split(',')
    user_df = pd.DataFrame([user_input], columns=df.columns.drop(target_variable) if data_type != 'clustering' else df.columns)

 
    if data_type == 'clustering':
        user_input_processed, _ = prepare_for_clustering(user_df)
    else:
        user_input_processed, _, _ = prepare_for_model(user_df, target_variable)

    prediction = model.predict(user_input_processed)
    if data_type == 'clustering':
        return f'Predicted cluster:{prediction[0]}'
    else:
        return f'Predicted {target_variable}: {prediction[0]}'

