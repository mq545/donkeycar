import math
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, cast

from donkeycar.parts.tflite import keras_model_to_tflite
from donkeycar.pipeline.sequence import TubRecord
from donkeycar.pipeline.sequence import TubSequence
from donkeycar.pipeline.types import TubDataset
from donkeycar.pipeline.augmentations import ImageAugmentation
from donkeycar.utils import get_model_by_type, normalize_image
import tensorflow as tf


class BatchSequence(object):
    """
    The idea is to have a shallow sequence with types that can hydrate
    themselves to np.ndarray initially and later into the types required by
    tf.data (i.e. dictionaries or np.ndarrays.
    """

    def __init__(self, model, config, records: List[TubRecord], is_train: bool):
        self.model = model
        self.config = config
        self.sequence = TubSequence(records)
        self.batch_size = self.config.BATCH_SIZE
        self.is_train = is_train
        self.augmentation = ImageAugmentation(config)
        self.pipeline = self.sequence.build_pipeline(
                x_transform=self.model.x_transform,
                y_transform=self.model.y_transform)
        self.pipeline_out = self._make_pipeline()
        self.types = self.model.output_types()
        self.shapes = self.model.output_shapes()

    def __len__(self):
        return math.ceil(len(self.pipeline_out) / self.batch_size)

    def _make_pipeline(self):
        """ This can be overridden if more complicated pipelines are
            required """
        # 1. We use image augmentation, it might do nothing, if there is no
        # augmentation in the config. It also does nothing in validation.
        # This works on uint8 images.
        pipeline_augment = self.sequence.map_pipeline(
            pipeline=self.pipeline,
            x_transform=lambda x: self.augmentation.augment(x) if
                                  self.is_train else x,
            y_transform=lambda y: y)

        # 2. we scale images to normalised float64 to be model inputs
        pipeline_normalise = list(self.sequence.map_pipeline(
            pipeline=pipeline_augment,
            x_transform=lambda x: normalize_image(x),
            y_transform=lambda y: y))

        return pipeline_normalise

    def make_tf_data(self):
        dataset = tf.data.Dataset.from_generator(
            generator=lambda: self.pipeline_out,
            output_types=self.types,
            output_shapes=self.shapes)
        return dataset.repeat().batch(self.batch_size)


def train(cfg, tub_paths, model, model_type):
    """
    Train the model
    """
    model_name, model_ext = os.path.splitext(model)
    is_tflite = model_ext == '.tflite'
    if is_tflite:
        model = f'{model_name}.h5'

    if not model_type:
        model_type = cfg.DEFAULT_MODEL_TYPE

    tubs = tub_paths.split(',')
    tub_paths = [Path(os.path.expanduser(tub)).absolute().as_posix() for tub in
                 tubs]
    output_path = os.path.expanduser(model)

    if 'linear' in model_type:
        train_type = 'linear'
    else:
        train_type = model_type

    kl = get_model_by_type(train_type, cfg)
    if cfg.PRINT_MODEL_SUMMARY:
        print(kl.model.summary())

    dataset = TubDataset(cfg, tub_paths)
    training_records, validation_records = dataset.train_test_split()
    print('Records # Training %s' % len(training_records))
    print('Records # Validation %s' % len(validation_records))

    training_pipe = BatchSequence(kl, cfg, training_records, is_train=True)
    validation_pipe = BatchSequence(kl, cfg, validation_records, is_train=False)

    dataset_train = training_pipe.make_tf_data()
    dataset_validate = validation_pipe.make_tf_data()
    train_size = len(training_pipe)
    val_size = len(validation_pipe)

    assert val_size > 0, "Not enough validation data, decrease the batch " \
                         "size or add more data."

    history = kl.train(model_path=output_path,
                       train_data=dataset_train,
                       train_steps=train_size,
                       batch_size=cfg.BATCH_SIZE,
                       validation_data=dataset_validate,
                       validation_steps=val_size,
                       epochs=cfg.MAX_EPOCHS,
                       verbose=cfg.VERBOSE_TRAIN,
                       min_delta=cfg.MIN_DELTA,
                       patience=cfg.EARLY_STOP_PATIENCE)

    if is_tflite:
        tflite_model_path = f'{os.path.splitext(output_path)[0]}.tflite'
        keras_model_to_tflite(output_path, tflite_model_path)

    return history
