from UnetModel import UnetModel
from UnetGenerator import UnetGenerator
from PlotLearning import PlotLearning

import time

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)


# configuration variables
input_size = 256
n_channels = 6
model_save_path = time.strftime("%Y%m%d-%Hh%Mm%Ss") + '_model.h5'

num_layers = 2
input_shape = (input_size, input_size, n_channels)

filters = 24
upconv_filters = 32

# model = Model(inputs=[inputs], outputs=[outputs])
model = UnetModel(
    input_shape=input_shape,
    filters=filters,
    upconv_filters=upconv_filters,
    num_layers=num_layers,
    )
model.summary()

pl = PlotLearning()
callbacks = [
    EarlyStopping(monitor="val_loss", patience=20,
                  verbose=1, mode="auto"),
    ModelCheckpoint(filepath=model_save_path,
                    verbose=1, save_best_only=True),
    pl,
]

model.compile(
    optimizer=Adam(
                learning_rate=0.0001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                amsgrad=False,
            ),
    loss='binary_crossentropy',
    metrics=["accuracy"]
)

train_generator = UnetGenerator(data_path='data/output/train/class1/', batch_size=4)
val_generator = UnetGenerator(data_path='data/output/val/class1/', batch_size=4)

results = model.fit_generator(
    train_generator,
    epochs=2,
    steps_per_epoch=5,
    validation_data=val_generator,
    callbacks=callbacks,
    validation_steps=5,
    verbose=1
)