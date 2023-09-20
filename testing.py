from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import json


def process_test_data(df, x_label, y_label=None):
    """
    Splits test data into images and labels. If y_label is not specified then only images are outputted.

    :param df: DataFrame
    :param x_label: str # maybe could handle List[str]
    :param y_label: List[str]
    :return: np.ndarray
    """
    image_df = df[x_label]
    images = []
    for _, img_path in image_df.items():
        with Image.open(img_path) as img:
            img = np.array(img)  # type: ignore
            img = img / 255.
        images.append(img)

    images_array = np.array(images)
    if y_label:
        label_df = df[y_label]
        labels_array = label_df.to_numpy()
        return images_array, labels_array

    return images_array


def plot_error_vs_truth(prediction_arr, truth_arr, labels, allowances, output_path):
    """

    Plots scatter plot with error (b/w predicted and truth values) on y-axis and truth values
    on x-axis. Horizontal red lines indicated error allowance for each label.

    """
    n_plots = len(labels)
    fig, axs = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4), tight_layout=True)
    for prediction, truth, label, allowance, ax in zip(prediction_arr.T, truth_arr.T, labels, allowances, axs):
        error = np.subtract(prediction, truth)
        ax.scatter(truth, error)
        ax.axhline(y=allowance, color='r', linestyle='-')
        ax.axhline(y=-allowance, color='r', linestyle='-')
        ax.set_title(label)
    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Truth")
    plt.ylabel("Error")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "error_vs_truth.png"))


def plot_truth_and_pred(prediction_arr, truth_arr, labels, output_path):
    """
    Plots truth values (blue) and predicted values (red).
    The x-axis is the observation number and y-axis is the value of either predicted or truth value.

    """
    n_plots = len(labels)
    fig, axs = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4), tight_layout=True)
    for prediction, truth, label, ax in zip(prediction_arr.T, truth_arr.T, labels, axs):
        ax.scatter(x=range(0, truth.size), y=truth, c='blue', label='Actual', alpha=0.3)
        ax.scatter(x=range(0, prediction.size), y=prediction, c='red', label='Predicted', alpha=0.3)
        ax.set_ylabel(label)

    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Observations")
    plt.title("Actual and Predicted Values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "truth_and_pred.png"))


def plot_error(prediction_arr, truth_arr, labels, allowances, output_path):
    """
    Plots a histogram of error b/w prediction and truth.
    Count is in the y-axis and error is in the x-axis.
    Red vertical line is error allowance and green vertical line is 2sigma value.

    """
    error_arr = prediction_arr - truth_arr
    shear_error, clock_error = np.sqrt(error_arr[:, 0] ** 2 + error_arr[:, 1] ** 2), error_arr[:, 2]
    comb_error_arr = np.vstack([shear_error, clock_error]).T
    num_runs = len(comb_error_arr)

    n_plots = len(labels)
    fig, axs = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4), tight_layout=True, sharex='col', sharey='row')
    for error, label, allowance, ax in zip(comb_error_arr.T, labels, allowances, axs):
        two_sigma = 2 * np.sqrt(np.mean(np.square(error)))  # 2 stds
        ax.hist(error, bins=40)
        measurement = "mm"
        if "clock" in label.lower():
            ax.axvline(x=-allowance, color='r', linestyle='-')
            ax.axvline(x=-two_sigma, color='g', linestyle=':')
            measurement = "mrad"

        ax.axvline(x=allowance, color='r', linestyle='-')
        ax.axvline(x=two_sigma, color='g', linestyle=':')
        ax.set_title(f'Prediction Errors - 2σ = {two_sigma:.3g} {measurement}')
        percent_marginal_error = ((allowance - two_sigma) / allowance) * 100
        ax.set_xlabel(f"{label} -- Margin of Error = {percent_marginal_error:.3g} %")

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    model_name = os.path.basename(os.path.normpath(output_path))
    plt.suptitle(f"{model_name} (total = {num_runs} runs)")
    plt.ylabel('Count')
    plt.xlabel('Error')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "error.png"))


def calc_metrics(prediction_arr, truth_arr, labels, output_path):
    """

    Calculates the MAE, MAPE, and RMSE of the error of the pred and truth.
    Metrics are stored in JSON format.

    """
    results = {}

    for prediction, truth, label in zip(prediction_arr.T, truth_arr.T, labels):
        metrics = {"MAE": round(mean_absolute_error(truth, prediction), 3),
                   "MAPE": round(np.mean((np.abs(np.subtract(truth, prediction) / truth))) * 100, 3),
                   "RMSE": round(np.sqrt(np.mean(np.square(prediction - truth))), 3)}

        results[label] = metrics

    with open(os.path.join(output_path, 'metric.json'), 'w') as file:
        json.dump(results, file)  # use `json.loads` to do the reverse


def plot_results(prediction_arr, truth_arr, labels, allowances, output_path):
    """
    Calls plot_error_vs_truth, plot_truth_and_pred, plot_error, calc_metrics

    :param prediction_arr: np.ndarray
    :param truth_arr: np.ndarray
    :param labels: List[str]
    :param allowances: List[float] # in the order of labels
    :param output_path: str
    :return: None
    """
    plot_error_vs_truth(prediction_arr, truth_arr, labels, allowances, output_path)
    plot_truth_and_pred(prediction_arr, truth_arr, labels, output_path)
    plot_error(prediction_arr, truth_arr, labels, allowances, output_path)
    calc_metrics(prediction_arr, truth_arr, labels, output_path)
