import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


def create_dataset(X, Y, lookback, out_steps):
    """
    Creates a TensorFlow time series dataset from the given input and output data.

    Parameters:
    X (pd.DataFrame or np.ndarray): The input features with shape (samples, lookback * features).
    Y (pd.DataFrame or np.ndarray): The target values with shape (samples, forecast).

    Returns:
    tf.data.Dataset: A TensorFlow dataset containing batches of the input and output data.
    """
    num_features = X.shape[1] // lookback

    if num_features == 0:
        X = X.values.reshape(X.shape[0], X.shape[1], 1)
    else:
        new_X = np.zeros((X.shape[0], lookback, num_features))
        for i in range(lookback):
            new_X[:, i, :] = X.iloc[:, i * num_features : (i + 1) * num_features].values
        X = new_X

    Y = Y.values.reshape(-1, out_steps, 1)

    # create batches of size 32
    ds = tf.data.Dataset.from_tensor_slices((X, Y)).batch(32)

    return ds


def feature_engineering(df, to_forecast, lookback, out_steps):
    """
    Performs feature engineering on the given DataFrame by creating lag features and future target values.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the features and target variable.
    to_forecast (str): The name of the target variable to forecast.

    Returns:
    tuple: A tuple containing the transformed DataFrame and the original columns.
    """
    columns = df.columns

    # create lag features
    for i in range(1, lookback):
        for f in columns:
            df[f"{f}_lag{i}"] = df[f].shift(i).values

    # create yhat for the next 48 hours
    for i in range(1, out_steps + 1):
        df[f"{to_forecast}_future{i}"] = df[to_forecast].shift(-i).values

    # drop rows with NaN values
    df.dropna(inplace=True)

    df.index = pd.RangeIndex(len(df.index))

    return df, columns


def pd_train_preprocessing(df_arr, to_forecast, lookback, out_steps):
    """
    Preprocesses the training data by performing feature engineering and scaling;
    Supports global datasets splitted across multiple files.

    Parameters:
    df_arr (list of pd.DataFrame): A list of DataFrames containing the training data.
    to_forecast (str): The name of the target variable to forecast.

    Returns:
    tuple: A tuple containing the preprocessed DataFrame and the scaler used for scaling.
    """
    # create a dataframe to then concatenate all the dataframes
    res = pd.DataFrame()

    for df in df_arr:
        df, columns = feature_engineering(df, to_forecast)
        res = pd.concat([res, df])

    scaler = StandardScaler()
    res = scaler.fit_transform(res)

    print(f"Feature engineering done for {to_forecast}, shape: {res.shape}")

    lag_features = [f"{f}_lag{i}" for i in range(1, lookback) for f in columns]
    future_features = [f"{to_forecast}_future{i}" for i in range(1, out_steps + 1)]

    all_columns = columns.tolist() + lag_features + future_features

    return pd.DataFrame(res, columns=all_columns), scaler


def pd_test_preprocessing(df, to_forecast, lookback, out_steps, scaler=None):
    """
    Preprocesses the test data by performing feature engineering and scaling.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the test data.
    to_forecast (str): The name of the target variable to forecast.
    scaler (StandardScaler, optional): The scaler used for scaling the training data. If None, a new scaler will be created.

    Returns:
    tuple: A tuple containing the preprocessed DataFrame and the scaler used for scaling.
    """
    df, columns = feature_engineering(df, to_forecast)

    # reuse scaler if provided
    if scaler is None:
        scaler = StandardScaler()
        df = scaler.fit_transform(df)
    else:
        df = scaler.transform(df)

    print(f"Feature engineering done for {to_forecast}, shape: {df.shape}")

    lag_features = [f"{f}_lag{i}" for i in range(1, lookback) for f in columns]
    future_features = [f"{to_forecast}_future{i}" for i in range(1, out_steps + 1)]

    all_columns = columns.tolist() + lag_features + future_features

    return pd.DataFrame(df, columns=all_columns), scaler