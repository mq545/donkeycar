import math
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, cast

from donkeycar.parts.tflite import keras_model_to_tflite
from donkeycar.pipeline.sequence import TubRecord
from donkeycar.pipeline.sequence import TubSequence as PipelineSequence
from donkeycar.pipeline.types import TubDataset
from donkeycar.utils import get_model_by_type
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import Sequence


class BatchSequence(object):
    # The idea is to have a shallow sequence with types that can hydrate
    # themselves to an ndarray

    def __init__(self, model, config, records: List[TubRecord] = list()):
        self.model = model
        self.config = config
        self.sequence = PipelineSequence(records)
        self.batch_size = self.config.BATCH_SIZE

        self.pipeline = list(self.sequence.build_pipeline(
                x_transform=self.model.x_transform,
                y_transform=self.model.y_transform))
        self.types = self.model.output_types()
        self.shapes = self.model.output_shapes()

    def __len__(self):
        return math.ceil(len(self.pipeline) / self.batch_size)

    def make_tf_data(self):
        dataset = tf.data.Dataset.from_generator(
            generator=lambda: self.pipeline,
            output_types=self.types,
            output_shapes=self.shapes)
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

    training_pipe = BatchSequence(kl, cfg, training_records)
    validation_pipe = BatchSequence(kl, cfg, validation_records)

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
