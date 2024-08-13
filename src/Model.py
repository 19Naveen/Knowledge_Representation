import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
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
    if data_type == 'classification':
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
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return X, y, preprocessor, le


def prediction_model(df, target_variable, data_type, user_input):
    """
    Train a prediction model, evaluate it, and make predictions on user input.
    
    Args:
    df (pandas.DataFrame): The input dataframe.
    target_variable (str): The name of the target variable column.
    data_type (str): The type of problem ('classification' or 'regression').
    user_input (str): Comma-separated string of user input for prediction.
    
    Returns:
    str: A string containing the model evaluation metrics and the prediction for the user input.
    """
    X, y, preprocessor, le = prepare_pipeline(df, target_variable)
    
    size = len(df)
    model = model_selection(data_type, size)
    
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    full_pipeline.fit(X_train, y_train)    
    y_pred = full_pipeline.predict(X_test)
    
    if data_type == 'classification':
        print('Accuracy:', accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred, target_names=le.classes_ if le else None))
    elif data_type == 'regression':
        print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
        print('R-squared:', r2_score(y_test, y_pred))
    
    user_input_list = user_input.split(',')
    user_df = pd.DataFrame([user_input_list], columns=X.columns)
    user_prediction = full_pipeline.predict(user_df)
    user_prediction = np.round(user_prediction).astype(int)
    
    if data_type == 'classification' and le:
        user_prediction = le.inverse_transform(user_prediction)
    result = f'Predicted {target_variable}: {user_prediction[0]}'
    
    if le:
        label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        result += '\n\nLabel Mapping:'
        for num, label in label_mapping.items():
            result +=  f'\n{num}: {label}'
    
    return result