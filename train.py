import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import tensorflow as tf
import os
import numpy as np


# TODO: Create a custom loss function that penalizes predictions within allowance

def split_data(input_path, test_size, valid_size):
    """
    Splits data df into train, val, test sets.
    Indicate test size and valid size as a decimal of percentage.

    :param input_path: str
    :param test_size: float
    :param valid_size: float
    :return: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    data = pd.read_csv(input_path)
    train, test = train_test_split(data, test_size=test_size, random_state=1)
    train, val = train_test_split(train, test_size=valid_size, random_state=1)
    print("shape train: ", train.shape)
    print("shape val: ", val.shape)
    print("shape test: ", test.shape)

    return train, val, test


def add_noise(image):
    """

    Adds Gaussian noise to image.

    :param image: np.ndarray
    :return: np.ndarray
    """
    gaussian = np.random.normal(0, 0.1 ** 0.5, image.shape)
    noisy_image = image + gaussian
    np.clip(noisy_image, 0., 255.)
    return image


class CustomImageDataGen(tf.keras.utils.Sequence):
    """
    Creates a custom image data generator to replace Keras's ImageDataGenerator.

    Reference:
    https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
    """

    def __init__(self, df, X_label, y_label,
                 batch_size,
                 input_size,
                 rescale=False,
                 shuffle=False):
        self.df = df.copy()
        self.X_col = X_label
        self.y_col = y_label
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.rescale = rescale
        self.n = len(self.df)

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __get_input(self, path):
        image = Image.open(path)
        image_arr = np.array(image)  # type: ignore

        if self.rescale:
            image_arr = image_arr / 255.

        return image_arr

    def __get_output(self, label):
        return np.array(label)

    def __get_data(self, batches):
        # Generates data containing batch_size samples

        path_batch = batches[self.X_col]
        label_batch = batches[self.y_col]

        X_batch = np.asarray([self.__get_input(x) for x in path_batch])
        y_batch = np.asarray([self.__get_output(y) for _, y in label_batch.iterrows()])
        return X_batch, y_batch

    def __getitem__(self, index):
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y

    def __len__(self):
        return self.n // self.batch_size


def create_generators(train, val, x_label, y_label, img_dim, batch_size, rescale=True, shuffle=True):
    """

    Creates training and validation image generators using the CustomImageDataGen.

    :param train: DataFrame
    :param val: DataFrame
    :param x_label: str
    :param y_label: List[str]
    :param img_dim: Tuple[int, int] # (height, width)
    :param batch_size: int
    :param rescale: bool
    :param shuffle: bool
    :return: Tuple[CustomImageDataGen, CustomImageDataGen]

    """
    train_generator = CustomImageDataGen(train,
                                         X_label=x_label,
                                         y_label=y_label,
                                         batch_size=batch_size,
                                         input_size=img_dim[:2],
                                         rescale=rescale,
                                         shuffle=shuffle)

    valid_generator = CustomImageDataGen(val,
                                         X_label=x_label,
                                         y_label=y_label,
                                         batch_size=batch_size,
                                         input_size=img_dim[:2],
                                         rescale=rescale,
                                         shuffle=shuffle)

    return train_generator, valid_generator


def get_callbacks(model_name, output_path):
    """

    Implements relevant callbacks.
    For a full list of possible callbacks refer to this:
    https://medium.com/red-buffer/callbacks-in-tensorflow-customize-the-behavior-of-your-training-a7b4f6e1cac2

    :param model_name: str
    :param output_path: str
    :return: List[tf.Callback, tf.Callback]
    """
    # stops training early if it does not see an improvement of min_delta within the patience value (epochs)
    early_stopping_callback = EarlyStopping(
        monitor="val_mean_absolute_percentage_error",
        min_delta=0.1,  # model should improve by at least 0.1%
        patience=10,  # amount of epochs with improvements worse than 0.1% until the model stops
        verbose=2,
        mode="min",
        restore_best_weights=True,  # restore the best model with the lowest validation error
    )
    # reduces learning rate by a factor every number of patience
    reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                           patience=5, min_lr=0.0001, verbose=1)

    # save a model if the monitor metric is better than previous epoch
    model_path = os.path.join(output_path, "models")
    os.makedirs(model_path, exist_ok=True)
    model_checkpoint_callback = ModelCheckpoint(
        f"{model_path}/{model_name}",
        monitor="val_mean_absolute_percentage_error",
        verbose=0,
        save_best_only=True,  # save the best model
        mode="min",
        save_freq="epoch",  # save every epoch
    )
    return [model_checkpoint_callback]


def run_model(model_name, model, train, valid, metrics, epochs, lr, output_path):
    """
    Compiles and trains the given model.

    :param model_name: str
    :param model: keras.Sequential
    :param train: CustomImageDataGen
    :param valid: CustomImageDataGen
    :param metrics: Dict[str: tf.Loss]
    :param epochs: int
    :param lr: float
    :param output_path: str
    :return: List[tf.History, tf.Sequential]
    """
    callbacks = get_callbacks(model_name, output_path)
    optimizer = Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer, loss=list(metrics.keys())[0], metrics=list(metrics.values()))

    history = model.fit(
        train,
        epochs=epochs,
        validation_data=valid,
        callbacks=callbacks,
        workers=8)

    model.summary()

    return history, model


def plot_history(histories, model_names, metrics, output_path):
    """

    Plots training and validation loss over a number of epochs.

    :param histories: List[tf.History]
    :param model_names: List[str]
    :param metrics: List[str] # just metric names
    :param output_path: str
    :return: None
    """

    for metric in metrics:
        history_metrics = []
        for model_history, model_name in zip(histories, model_names):
            train_metric = {'metric': model_history.history[metric],
                            'type': 'training',
                            'model': model_name}
            valid_metric = {'metric': model_history.history[f'val_{metric}'],
                            'type': 'validation',
                            'model': model_name}

            s1 = pd.DataFrame(train_metric)
            history_metrics.append(s1)
            s2 = pd.DataFrame(valid_metric)
            history_metrics.append(s2)

        df = pd.concat(history_metrics, axis=0).reset_index()

        grid = sns.relplot(data=df, x=df["index"], y="metric", hue="model", col="type", kind="line", legend=True)
        grid.set(ylabel=metric)
        for ax in grid.axes.flat:
            ax.set(xlabel="Epoch")
        plt.savefig(os.path.join(output_path, f"train_valid_{metric}.png"))
