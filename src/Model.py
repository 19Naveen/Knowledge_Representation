from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold


def classification_model_selection(size):
    """
    Selects a classification model based on the given size.

    Parameters:
    - size (int): The size of the dataset.

    Returns:
    - model: The selected classification model.

    """
    if size <= 1000:
        return SVC()
    elif size <= 10000:
        return RandomForestClassifier()
    else:
        return XGBClassifier()


def regression_model_selection(size):
    """
    Selects a regression model based on the given size.

    Parameters:
        size (int): The size of the dataset.

    Returns:
        model: The selected regression model.

    """
    if size <= 1000:
        return LinearRegression()
    elif size <= 10000:
        return RandomForestRegressor()
    else:
        return XGBRegressor()


def model_selection(data_type, size):
    """
    Selects a model based on the given data type and size.

    Parameters:
    - data_type (str): The type of data ('classification' or 'regression').
    - size (int): The size of the data.

    Returns:
    - The selected model based on the data type and size.
    """
    if data_type == 'classification':
        return classification_model_selection(size)
    else:
        return regression_model_selection(size)
    
    

def prepare_for_model(df, target_variable, pca_threshold=0.95, variance_threshold=0.01):
    """
    Prepare the dataframe for model training.
    
    Args:
    df (pd.DataFrame): Input dataframe
    target_variable (str): Name of the target variable column
    pca_threshold (float): Explained variance ratio threshold for PCA
    variance_threshold (float): Threshold for variance-based feature selection
    
    Returns:
    X (np.array): Prepared feature matrix
    y (np.array): Target variable
    feature_names (list): List of feature names after preprocessing
    """
    
    X = df.drop(columns=[target_variable])
    y = df[target_variable].values
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
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


def prediction_model(df, target_variable, data_type, user_input):
    """
    Predicts the target variable using a machine learning model.
    Args:
        df (pandas.DataFrame): The input dataframe.
        target_variable (str): The name of the target variable column.
        data_type (str): The type of the problem, either 'classification' or 'regression'.
        user_input (str): The user input for prediction.
    Returns:
        str: The predicted value of the target variable.
    """
    X_selected, y, feature_names = prepare_for_model(df, target_variable)
    size = len(df)
    model = model_selection(data_type, size)
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if data_type == 'classification':
        print('Accuracy:', accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))
    elif data_type == 'regression':
        print('Mean Absolute Error:', np.mean(np.abs(y_pred - y_test)))
        print('Mean Squared Error:', np.mean((y_pred - y_test)**2))
        print('R-squared:', model.score(X_test, y_test))
    
    # Convert user input to DataFrame
    user_input_list = user_input.split(',')
    user_df = pd.DataFrame([user_input_list], columns=df.drop(columns=[target_variable]).columns)
    
    # Prepare user input
    user_X, _, _ = prepare_for_model(user_df, target_variable)
    
    # Make prediction
    prediction = model.predict(user_X)
    return f'Predicted {target_variable}: {prediction[0]}'