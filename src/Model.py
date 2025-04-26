import numpy as np
import pandas as pd
from sklearn.svm import SVC
import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score

def classification_model_selection(size):
    """
    Select a classification model based on the dataset size.
    
    Args:
    size (int): The number of samples in the dataset.
    
    Returns:
    sklearn estimator: A classification model suitable for the given dataset size.
    """
    if size <= 1000:
        return SVC(probability=True)
    elif size <= 10000:
        return RandomForestClassifier()
    else:
        return XGBClassifier()


def regression_model_selection(size):
    """
    Select a regression model based on the dataset size.
    
    Args:
    size (int): The number of samples in the dataset.
    
    Returns:
    sklearn estimator: A regression model suitable for the given dataset size.
    """
    if size <= 1000:
        return LinearRegression()
    elif size <= 10000:
        return RandomForestRegressor()
    else:
        return XGBRegressor()


def model_selection(data_type, size):
    """
    Select a model based on the problem type and dataset size.
    
    Args:
    data_type (str): The type of problem ('classification' or 'regression').
    size (int): The number of samples in the dataset.
    
    Returns:
    sklearn estimator: A model suitable for the given problem type and dataset size.
    """
    if data_type == 'Classification':
        return classification_model_selection(size)
    else:
        return regression_model_selection(size)


def prepare_pipeline(df, target_variable):
    """
    Prepare the data pipeline for model training.
    
    Args:
    df (pandas.DataFrame): The input dataframe.
    target_variable (str): The name of the target variable column.
    
    Returns:
    tuple: X (features), y (target), preprocessor (ColumnTransformer), le (LabelEncoder if applicable)
    """
    X = df.drop(columns=[target_variable])
    y = df[target_variable]

    le = None
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
        #,('pca', PCA(n_components=0.95))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return X, y, preprocessor, le


def create_model(df, target_variable, data_type):
    """
    Train a prediction model, evaluate it, and make predictions on user input.
    
    Args:
    df (pandas.DataFrame): The input dataframe.
    target_variable (str): The name of the target variable column.
    data_type (str): The type of problem ('classification' or 'regression').
    user_input (str): Comma-separated string of user input for prediction.
    
    Returns:
    Accuracy, le: A string containing the model evaluation metrics and labelEncoder.
    """
    X, y, preprocessor, le = prepare_pipeline(df, target_variable)
    data_type = data_type.lower()
    size = len(df)
    model_accuracy = 'Information Unavailable at this Moment'
    model = model_selection(data_type, size)
    
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    full_pipeline.fit(X_train, y_train)   
    with open("./model/my_pipeline.pkl", "wb") as f:
        pickle.dump(full_pipeline, f)

    y_pred = full_pipeline.predict(X_test)
    if data_type == 'classification':
        accuracy = accuracy_score(y_test, y_pred)
        model_accuracy = f"Accuracy Score: {accuracy}"
    elif data_type == 'regression':
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        model_accuracy = f"Mean Absolute Error: {mae}\nMean Squared Error: {mse}\nR-squared: {r2}"
    else:
        raise ValueError(f"Invalid data_type. Expected 'classification' or 'regression' but got {data_type}.")
    
    return model_accuracy, le

def predict_model(user_input, column_dropped, columns, data_type, le):
    data_type = data_type.lower()
    with open("./model/my_pipeline.pkl", "rb") as f:
        full_pipeline = pickle.load(f)

    user_input_list = list(user_input.values())
    user_df = pd.DataFrame([user_input_list], columns=columns)
    user_prediction = full_pipeline.predict(user_df)
    user_prediction = np.round(user_prediction).astype(int)
    
    if data_type == 'classification' and le:
        user_prediction = le.inverse_transform(user_prediction)
    result = f'Predicted {column_dropped}: {user_prediction[0]}'
    
    if le:
        label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        result += '\n\nLabel Mapping:'
        for num, label in label_mapping.items():
            result +=  f'\n{num}: {label}'
    
    return result