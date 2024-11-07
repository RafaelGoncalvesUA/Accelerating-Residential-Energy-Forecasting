import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_history(history, to_forecast):
    """
    Plots the training and validation loss history.

    Parameters:
    history (tf.keras.callbacks.History): The history object returned by the fit method of a Keras model.
    """
    plt.figure(figsize=(7, 5))
    plt.plot(history.history["loss"], label="train")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="test")
    plt.legend()
    plt.show()
    # plt.savefig(f'images/loss_curves/{model_name}_{to_forecast}.png')


def plot_residuals_distribution(y_true, y_pred, to_forecast):
    """
    Plots the distribution of residuals (errors) for specific forecast horizons.

    Parameters:
    y_true (pd.DataFrame or np.ndarray): The true values of the target variable.
    y_pred (pd.DataFrame or np.ndarray): The predicted values of the target variable.
    to_forecast (str): The name of the target variable being forecasted.

    Returns:
    pd.DataFrame: A DataFrame containing the residuals (errors) for each forecast horizon.
    """
    errors = y_pred - y_true
    errors_plot = errors[
        [
            f"{to_forecast}_future1",
            f"{to_forecast}_future10",
            f"{to_forecast}_future20",
            f"{to_forecast}_future30",
            f"{to_forecast}_future40",
        ]
    ]
    errors_plot = np.abs(errors_plot)

    # rename columns for plotting
    errors_plot.columns = ["1h", "10h", "20h", "30h", "40h"]

    sns.violinplot(errors_plot)
    plt.show()
    return errors
