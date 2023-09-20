import pandas as pd

from model import SimpleModels, PretrainedModels
from train import split_data, create_generators, run_model, plot_history
from testing import process_test_data, plot_results
import os
from PIL import Image
import numpy as np
from tensorflow.python.keras.losses import MeanAbsoluteError, MeanAbsolutePercentageError
from tensorflow.python.keras.metrics import RootMeanSquaredError
from sklearn.preprocessing import StandardScaler


# Data Paths
dir = "trial_results"
input_path = os.path.join(dir, "data.csv")  # replace input_path w path/to/dir/data.csv
output_path = dir  # replace output_path w path/to/dir

# Hyper-parameters
TEST_SIZE, VALID_SIZE = 0.2, 0.2

# experiment to see changes in performance
EPOCHS = 100  # usually multiples for 50 (50 - 200)
BATCH_SIZE = 24  # usually multiples of 2 (16, 24, 32)
LR = 0.001  # typically powers of 10 (0.01 - 0.0001)

X_LABEL = "image"
Y_LABEL = ["MASK_dx", "MASK_dy", "MASK_dclock"]

NUM_OUTPUT = len(Y_LABEL)
DSHEAR_error, DCLOCK_error = 0.12, 2.0

sample_df = pd.read_csv(input_path)
IMG_ARR = np.array(Image.open(sample_df[X_LABEL].iloc[0]))  # type: ignore
IMG_DIM = IMG_ARR.shape
print(f"Image Shape: {IMG_DIM}")

simple_model = SimpleModels(IMG_DIM, NUM_OUTPUT)
pretrained_model = PretrainedModels(IMG_DIM, NUM_OUTPUT)


MODELS = {
    "model_conv_32_64_64": simple_model.model_32_64_64()
}

METRICS = {'mean_absolute_error': MeanAbsoluteError(),
           'mean_absolute_percentage_error': MeanAbsolutePercentageError(),
           'root_mean_squared_error': RootMeanSquaredError()}


def run(models, metrics, saved_model=False):
    """

    Trains and tests given models.
    If saved_model is True, the most recently saved model is used for testing.

    :param models: Dict[str: tf.Sequential]
    :param metrics: Dict[str: tf.Loss]
    :param saved_model: bool
    :return: None
    """
    print("splitting images...")
    train, val, test = split_data(input_path, TEST_SIZE, VALID_SIZE)

    scaler = StandardScaler()
    train[Y_LABEL] = scaler.fit_transform(train[Y_LABEL])

    if saved_model:
        model_weights_dir = os.path.join(os.path.dirname(input_path), "models")

        for model_name in models:
            model_weight_path = os.path.join(model_weights_dir, model_name)
            models[model_name].load_weights(model_weight_path).expect_partial()

        trained_models = models

    else:
        val[Y_LABEL] = scaler.transform(val[Y_LABEL])
        train_generator, valid_generator = create_generators(train, val, X_LABEL, Y_LABEL, IMG_DIM, BATCH_SIZE)

        model_names = list(models.keys())

        metric_names = list(metrics.keys())

        print("training")
        histories = []
        trained_models = {}
        for model_name, model in models.items():
            model_history, trained_model = run_model(
                model_name=model_name,
                model=model,
                train=train_generator,
                valid=valid_generator,
                metrics=metrics,
                epochs=EPOCHS,
                lr=LR,
                output_path=output_path)

            histories.append(model_history)
            trained_models[model_name] = trained_model

        print("plotting training results")
        train_results_path = os.path.join(output_path, "train_results")
        os.makedirs(train_results_path, exist_ok=True)
        plot_history(histories, model_names, metric_names, train_results_path)

    print("processing testing data")
    test_images, test_labels = process_test_data(test, X_LABEL, Y_LABEL)
    test_results_path = os.path.join(output_path, "test_results")

    print("testing")
    allowances = [DSHEAR_error, DCLOCK_error]
    labels = ["MASK_dshear", "MASK_dclock"]
    for model_name, trained_model in trained_models.items():
        test_results_model_path = os.path.join(test_results_path, model_name)
        os.makedirs(test_results_model_path, exist_ok=True)
        prediction, truth = trained_model.predict(test_images), test_labels
        prediction = scaler.inverse_transform(prediction)
        assert prediction.shape == truth.shape
        print(f"plotting testing results for {model_name}")
        plot_results(prediction, truth, labels, allowances, test_results_model_path)


run(models=MODELS, metrics=METRICS)
