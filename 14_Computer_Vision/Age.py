import os

import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam


def load_train(path):
    df = pd.read_csv(os.path.join(path, 'labels.csv'))
    train_datagen = ImageDataGenerator(
        validation_split=.25,
        rescale=1/255.,
        vertical_flip=True,
    )
    train_datagen_flow = train_datagen.flow_from_dataframe(
        dataframe=df,
        directory=os.path.join(path, 'final_files'),
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        seed=12345,
        subset='training',
    )

    return train_datagen_flow


def load_test(path):
    df = pd.read_csv(os.path.join(path, 'labels.csv'))
    test_datagen = ImageDataGenerator(
        validation_split=.25,
        rescale=1/255.,
    )
    test_datagen_flow = test_datagen.flow_from_dataframe(
        dataframe=df,
        directory=os.path.join(path, 'final_files'),
        target_size=(224, 224),
        x_col='file_name',
        y_col='real_age',
        batch_size=32,
        class_mode='raw',
        seed=12345,
        subset='validation',
    )

    return test_datagen_flow


def create_model(input_shape):
    backbone = ResNet50(
        input_shape=input_shape,
        weights='/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
        include_top=False,
        )

    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_absolute_error', optimizer=Adam(lr=.0001), metrics=['mae'])

    return model


def train_model(
        model,
        train_data,
        test_data,
        batch_size=None,
        epochs=10,
        steps_per_epoch=None,
        validation_steps=None,
):
    model.fit(
        train_data,
        validation_data=test_data,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=2,
        shuffle=True,
    )

    return model
