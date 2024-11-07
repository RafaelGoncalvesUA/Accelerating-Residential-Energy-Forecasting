from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import os
import pickle as pkl


class CustomPCA:
    """
    A custom PCA class to reduce the dimensionality of a dataset and save/load the number of principal components to keep.

    Attributes:
        filename (str): The filename to save/load the number of principal components to keep.
        threshold (float): The variance threshold to determine the number of components to keep.
    """

    def __init__(self, filename, var_threshold=0.95):
        self.filename = filename
        self.threshold = var_threshold

    def fit(self, df, to_forecast, out_steps):
        """
        Fits the PCA model to the dataframe and reduces its dimensionality.

        Parameters:
            df (pd.DataFrame): The input dataframe.
            to_forecast (str): The column name to forecast.
            out_steps (int): The number of future steps to forecast.

        Returns:
            pd.DataFrame: The dataframe with reduced dimensions.
        """
        columns_to_keep = [
            f"{to_forecast}_future{i}" for i in range(1, out_steps + 1)
        ] + ["Hour", "Day Type"]
        self.to_forecast = to_forecast
        self.out_steps = out_steps

        df_keep = df[columns_to_keep]
        df_reduce = df.drop(columns_to_keep, axis=1)

        if os.path.exists(self.filename):
            with open(self.filename, "rb") as f:
                n_components_to_keep = pkl.load(f)
        else:
            pca = PCA()
            pca.fit(df_reduce)
            n_components_to_keep = np.where(
                np.cumsum(pca.explained_variance_ratio_) > self.threshold
            )[0][0]

            with open(self.filename, "wb") as f:
                pkl.dump(n_components_to_keep, f)

        self.pca = PCA(n_components=n_components_to_keep)
        print(f"Keeping {n_components_to_keep} components")

        df_reduce = pd.DataFrame(
            self.pca.fit_transform(df_reduce),
            columns=[f"PC{i}" for i in range(n_components_to_keep)],
        )

        data = pd.concat([df_keep, df_reduce], axis=1)
        return data

    def predict(self, df):
        """
        Transforms the given DataFrame using the trained PCA model and returns the transformed DataFrame.

        Parameters:
        df (pd.DataFrame): The input DataFrame to be transformed.

        Returns:
        pd.DataFrame: The transformed DataFrame with PCA components and the original columns to keep.

        Raises:
        Exception: If the PCA model has not been trained yet.
        """
        if self.pca is None:
            raise Exception("PCA not trained yet")

        columns_to_keep = [
            f"{self.to_forecast}_future{i}" for i in range(1, self.out_steps + 1)
        ] + ["Hour", "Day Type"]

        df_keep = df[columns_to_keep]
        df_reduce = df.drop(columns_to_keep, axis=1)

        df_reduce = pd.DataFrame(
            self.pca.transform(df_reduce),
            columns=[f"PC{i}" for i in range(self.pca.n_components_)],
        )

        data = pd.concat([df_keep, df_reduce], axis=1)
        return data
