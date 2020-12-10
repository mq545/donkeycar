#!/usr/bin/env python3
"""
Scripts to train a keras model using tensorflow.
Basic usage should feel familiar: python train_v2.py --model models/mypilot

Usage:
    train.py [--tubs=tubs] (--model=<model>) [--type=(linear|inferred|tensorrt_linear|tflite_linear)]

Options:
    -h --help              Show this screen.
"""

import math
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, cast

import donkeycar
import numpy as np
from docopt import docopt
from donkeycar.parts.keras import KerasCategorical, KerasInferred, KerasLinear
from donkeycar.parts.tflite import keras_model_to_tflite
from donkeycar.parts.tub_v2 import Tub
from donkeycar.pipeline.sequence import TubRecord
from donkeycar.pipeline.sequence import TubSequence as PipelineSequence
from donkeycar.pipeline.types import TubDataset
from donkeycar.utils import get_model_by_type, linear_bin, train_test_split
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.utils.data_utils import Sequence


class BatchSequence(Sequence):
    # Improve batched_pipeline to make most of this go away as well.
    # The idea is to have a shallow sequence with types that can hydrate themselves to an ndarray

    def __init__(self, keras_model, config, records: List[TubRecord] = list()):
        self.keras_model = keras_model
        self.config = config
        self.records = records
        self.sequence = PipelineSequence(self.records)
        self.batch_size = self.config.BATCH_SIZE
        self.consumed = 0

        # Keep track of model type
        # Eventually move this part into the model itself.
        self.is_linear = type(self.keras_model) is KerasLinear
        self.is_inferred = type(self.keras_model) is KerasInferred
        self.is_categorical = type(self.keras_model) is KerasCategorical
        print("Pipeline model")
        print(self.is_linear, self.is_categorical, self.is_inferred)

        # Define transformations
        def x_transform(record: TubRecord):
            # Using an identity transform to delay image loading
            img_arr = record.image(cached=True, normalize=True)
            return img_arr

        def y_categorical(record: TubRecord):
            angle: float = record.underlying['user/angle']
            throttle: float = record.underlying['user/throttle']
            R = self.config.MODEL_CATEGORICAL_MAX_THROTTLE_RANGE
            angle = linear_bin(angle, N=15, offset=1, R=2.0)
            throttle = linear_bin(throttle, N=20, offset=0.0, R=R)
            return {'angle_out': angle, 'throttle_out': throttle}

        def y_inferred(record: TubRecord):
            angle: float = record.underlying['user/angle']
            return {'n_outputs0': angle}

        def y_linear(record: TubRecord):
            angle: float = record.underlying['user/angle']
            throttle: float = record.underlying['user/throttle']
            return {'n_outputs0': angle, 'n_outputs1': throttle}

        if self.is_linear:
            self.pipeline = list(self.sequence.build_pipeline(x_transform=x_transform, y_transform=y_linear))
            self.output_types = (tf.float64, {'n_outputs0': tf.float64,
                                              'n_outputs1': tf.float64})
        elif self.is_categorical:
            self.pipeline = list(self.sequence.build_pipeline(x_transform=x_transform, y_transform=y_categorical))
            self.output_types = (tf.float64, {'angle_out': tf.float64,
                                              'throttle_out': tf.float64})
        else:
            self.pipeline = list(self.sequence.build_pipeline(x_transform=x_transform, y_transform=y_inferred))
            self.output_types = (tf.float64, {'n_outputs0': tf.float64})

    def __len__(self):
        if not self.pipeline:
            raise RuntimeError('Pipeline is not initialized')

        return math.ceil(len(self.pipeline) / self.batch_size)

    def __getitem__(self, index):
        count = 0
        images = []
        angles = []
        throttles = []
        while count < self.batch_size:
            i = (index * self.batch_size) + count
            if i >= len(self.pipeline):
                break

            record, r = self.pipeline[i]
            images.append(record.image(cached=False, normalize=True))

            if isinstance(r, tuple):
                angle, throttle = r
                angles.append(angle)
                throttles.append(throttle)
            else:
                angles.append(r)

            count += 1

        X = np.array(images)
        if self.is_inferred:
            Y = np.array(angles)
        else:
            Y = [np.array(angles), np.array(throttles)]
        return X, Y

    def make_tf_data(self):
        dataset = tf.data.Dataset.from_generator(
            generator=lambda: self.pipeline,
            output_types=self.output_types)
        return dataset.repeat().batch(self.batch_size)


class ImagePreprocessing(Sequence):
    '''
    A Sequence which wraps another Sequence with an Image Augumentation.
    '''

    def __init__(self, sequence, augmentation):
        self.sequence = sequence
        self.augumentation = augmentation

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index):
        X, Y = self.sequence[index]
        return self.augumentation.augment_images(X), Y


def train(cfg, tub_paths, output_path, model_type):
    """
    Train the model
    """
    # convert single path into list of one element
    if type(tub_paths) is str:
        tub_paths = [tub_paths]

    if 'linear' in model_type:
        train_type = 'linear'
    else:
        train_type = model_type

    kl = get_model_by_type(train_type, cfg)
    kl.compile()

    if cfg.PRINT_MODEL_SUMMARY:
        print(kl.model.summary())

    batch_size = cfg.BATCH_SIZE
    dataset = TubDataset(cfg, tub_paths)
    training_records, validation_records = dataset.train_test_split()
    print('Records # Training %s' % len(training_records))
    print('Records # Validation %s' % len(validation_records))

    training_pipe = BatchSequence(kl, cfg, training_records)
    validation_pipe = BatchSequence(kl, cfg, validation_records)

    dataset_train = training_pipe.make_tf_data()
    dataset_validate = validation_pipe.make_tf_data()
    train_size = len(training_pipe)
    val_size = len(validation_pipe)

    assert val_size > 0, "Not enough validation data, decrease the " \
                                "batch size or add more data."

    # Setup early stoppage callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=cfg.EARLY_STOP_PATIENCE),
        ModelCheckpoint(
            filepath=output_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
        )
    ]

    history = kl.model.fit(
        x=dataset_train,
        steps_per_epoch=train_size,
        batch_size=batch_size,
        callbacks=callbacks,
        validation_data=dataset_validate,
        validation_steps=val_size,
        epochs=cfg.MAX_EPOCHS,
        verbose=cfg.VERBOSE_TRAIN,
        workers=1,
        use_multiprocessing=False
    )
    return history


def main():
    args = docopt(__doc__)
    cfg = donkeycar.load_config()
    tubs = args['--tubs']
    model = args['--model']
    model_type = args['--type']
    model_name, model_ext = os.path.splitext(model)
    is_tflite = model_ext == '.tflite'
    if is_tflite:
        model = f'{model_name}.h5'

    if not model_type:
        model_type = cfg.DEFAULT_MODEL_TYPE

    tubs = tubs.split(',')
    data_paths = [Path(os.path.expanduser(tub)).absolute().as_posix() for tub in tubs]
    output_path = os.path.expanduser(model)
    history = train(cfg, data_paths, output_path, model_type)
    if is_tflite:
        tflite_model_path = f'{os.path.splitext(output_path)[0]}.tflite'
        keras_model_to_tflite(output_path, tflite_model_path)


if __name__ == "__main__":
    main()
