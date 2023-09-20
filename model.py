from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D, \
    MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense, Reshape, Normalization, Resizing, Input
from tensorflow.keras.applications import EfficientNetB0, ResNet50


# TODO: implement Branched Model

class BranchedModel:
    """
    Branched Model should have a common layer that splits off to separate branches in order to
    test different subsets of variables (e.g. mask values, smst values, etc.).

    Note that this will most likely require changes to ImageDataGenerator to
    yield multiple labels (i.e. not in an array).

    Reference Material:
    https://towardsdatascience.com/building-a-multi-output-convolutional-neural-network-with-keras-ed24c7bc1178
    https://datascience.stackexchange.com/questions/82236/one-neural-network-with-multiple-outputs-or-multiple-neural-networks-with-a-sing
    https://blog.paperspace.com/combining-multiple-features-outputs-keras/

    """

    def __init__(self, input_shape, num_output):
        self.input_shape = input_shape
        self.num_shear_output, self.num_clock_output = num_output

    def common_layer(self, inputs):
        x = Conv2D(16, (3, 3), padding="same")(inputs)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)

        x = Dropout(0.25)(x)
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Dropout(0.25)(x)
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        return x

    def shear_branch(self, inputs):
        x = self.common_layer(inputs)
        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(self.num_shear_output, name="shear_output")(x)

        return x

    def clock_branch(self, inputs):
        x = self.common_layer(inputs)
        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(self.num_clock_output, name="clock_output")(x)

        return x

    def complete_model(self):
        inputs = Input(shape=self.input_shape)
        shear_branch, clock_branch = self.shear_branch(inputs), self.clock_branch(inputs)
        outputs = [shear_branch, clock_branch]
        model = Model(inputs=inputs,
                      outputs=outputs,
                      name="branch_net")

        return model


class PretrainedModels:
    """
    Utilizes previously trained weights of large architectures
    in order to finetune them in our training.

    """

    def __init__(self, input_shape, num_output):
        self.input_shape = input_shape
        self.num_output = num_output

    # poor performance and requires tensorflow downgrade -- do not use
    def model_efficient_net(self):
        model = Sequential()
        # input shapes of the images should always be 224x224x3
        model.add(Resizing(224, 224, input_shape=self.input_shape))
        base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
        base_model.trainable = False
        model.add(base_model)
        model.add(GlobalAveragePooling2D())
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Dense(self.num_output))

        return model

    # okay performance - not better than baseline_fc or conv_32_64_64
    def model_resnet_50(self):
        model = Sequential()
        # input shapes of the images should always be 224x224x3
        model.add(Resizing(224, 224, input_shape=self.input_shape))
        base_model = ResNet50(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
        base_model.trainable = False
        model.add(base_model)
        model.add(Flatten())
        model.add(Dense(100, activation="relu"))
        model.add(Dense(100, activation="relu"))
        model.add(Dense(100, activation="relu"))
        model.add(Dense(self.num_output))

        return model


class SimpleModels:
    """
    Custom models. Baseline models are implemented from Robert's code.

    Performance Ranking (as of 2/3/2023):
    1) model_32_64_64
    2) model_baseline_fc
    3) model_64_64_3
    4) model_baseline_conv
    """
    def __init__(self, input_shape, num_output):
        self.input_shape = input_shape
        self.num_output = num_output

    def model_baseline_fc(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dropout(0.25))
        model.add(Dense(50, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(20, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(self.num_output))

        return model

    def model_baseline_conv(self):
        model = Sequential()
        model.add(Conv2D(filters=8, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(AveragePooling2D(strides=(2, 2)))

        model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(AveragePooling2D(strides=(2, 2)))

        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())

        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(self.num_output))

        return model

    def model_32_64_64(self):
        model = Sequential()
        if len(self.input_shape) == 2:
            model.add(Reshape(self.input_shape + (1,), input_shape=self.input_shape))
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPool2D((2, 2)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPool2D((2, 2)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))

        model.add(Flatten())
        model.add(Dense(64, activation="relu"))
        model.add(Dense(self.num_output))

        return model

    def model_64_64_3(self):
        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                         input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(MaxPool2D(strides=(2, 2)))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(strides=(2, 2)))

        model.add(Conv2D(filters=3, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(self.num_output))

        return model
