import os
import numpy as np
from abc import ABC, abstractmethod
from tensorflow.python.keras.utils.data_utils import Sequence
from donkeycar.parts.tub_v2 import Tub
from donkeycar.utils import get_model_by_type, load_image_arr, \
    train_test_split, normalize_image

DEFAULT_TRANSFORMATIONS = ['ImageReader', 'ImageNormalizer']


class TubDataset(object):
    '''
    Loads the dataset, and creates a train/test split.
    '''

    def __init__(self, tub_paths, test_size=0.2, shuffle=True):
        self.tub_paths = tub_paths
        self.test_size = test_size
        self.shuffle = shuffle
        self.tubs = [Tub(tub_path, read_only=True) for tub_path in
                     self.tub_paths]
        self.records = list()

    def train_test_split(self):
        print('Loading tubs from paths %s' % self.tub_paths)
        for tub in self.tubs:
            for record in tub:
                record['_image_base_path'] = tub.images_base_path
                self.records.append(record)

        return train_test_split(self.records, shuffle=self.shuffle,
                                test_size=self.test_size)


class LazyRecord(object):
    """ Lazy record which wraps around record dictionary. There is no
        additional functionality here, it's all in the derived classes. """

    def __init__(self, record, transforms):
        self.record = record
        self.transforms = transforms

    def get_x_y(self, model):
        """ Dispatch returning of X, Y to the model which will call
            get_entry() on the LazyRecord and from there drive the
            transformations. """
        return model.get_x_y(self)

    def get_entry(self, key):
        """ Get entry from record and run through transformations """
        val = self.record.get(key)
        for transform in self.transforms:
            val = transform.get(self.record, key, val)

        return val

    def has_entry(self, key):
        """ Check if record available """
        return key in self.record

    def _set_entry(self, key, val):
        """ Set entry - should only be called in derived classes """
        self.record[key] = val


class RecordTransformer(ABC):
    """ Base class for record transformations which can be stacked on top of
        each other """

    def __init__(self, key, config, cache=False):
        """
        :param key:             key on which the transformation should apply
        :param cache:           if transformed record should be put back for
                                performance - this destroys the original record
        """
        self.config = config
        self.key = key
        self.cache = cache

    def get(self, record, key, val):
        """ Override the base class to allow transforming and caching of
            transformed entries. """
        if key == self.key:
            val_trans = self.transform(record, val)
            if self.cache:
                record[key] = val_trans
            return val_trans
        else:
            return val

    @abstractmethod
    def transform(self, record, val):
        """ This has to be implemented in derived classes"""
        pass

    @classmethod
    def create(cls, transform_names, config):
        """ Method to create a stack of transformations on top of the
            LazyRecord """
        transforms = [globals()[name](config) for name in transform_names]
        return transforms


class ImageReader(RecordTransformer):
    """ Convert path into image array """
    def __init__(self, config):
        super().__init__('cam/image_array', config, True)

    def transform(self, record, val):
        if type(val) is str:
            base_path = record['_image_base_path']
            # only transform once into numpy img, when value is path to image
            image_path = os.path.join(base_path, val)
            image = load_image_arr(image_path, self.config)
            return image
        else:
            return val


class ImageNormalizer(RecordTransformer):
    """ Normalize Images from np.uint8 to np.float32. We don't want to cache
        these as they require 4x memory. """
    def __init__(self, config):
        super().__init__('cam/image_array', config, False)

    def transform(self, record, val):
        return normalize_image(val)


class TubSequence(Sequence):
    def __init__(self, keras_model, config, records):
        self.keras_model = keras_model
        self.config = config
        self.records = records
        self.batch_size = self.config.BATCH_SIZE
        transforms = getattr(self.config, 'RECORD_TRANSFORMATIONS',
                             DEFAULT_TRANSFORMATIONS)
        self.transformations = RecordTransformer.create(transforms, self.config)

    def __len__(self):
        return len(self.records) // self.batch_size

    def __getitem__(self, index):
        count = 0
        records = []

        while count < self.batch_size:
            i = (index * self.batch_size) + count
            if i >= len(self.records):
                break

            record = LazyRecord(self.records[i], self.transformations)
            records.append(record)
            count += 1

        x = []
        y = []
        # collecting across the whole batch
        for record in records:
            single_x, single_y = record.get_x_y(self.keras_model)
            x.append(single_x)
            y.append(single_y)

        # reshape X, Y
        def reshape(z):
            dim = len(z[0])
            if dim == 1:
                return np.array([zi[0] for zi in z])
            else:
                ret_z = []
                for j in range(dim):
                    z_j = np.array([zi[j] for zi in z])
                    ret_z.append(z_j)
                return ret_z

        x_res = reshape(x)
        y_res = reshape(y)
        return x_res, y_res


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

    if cfg.PRINT_MODEL_SUMMARY:
        print(kl.model.summary())

    batch_size = cfg.BATCH_SIZE
    dataset = TubDataset(tub_paths, test_size=(1. - cfg.TRAIN_TEST_SPLIT),
                         shuffle=True)
    training_records, validation_records = dataset.train_test_split()
    print('Records # Training %s' % len(training_records))
    print('Records # Validation %s' % len(validation_records))

    training = TubSequence(kl, cfg, training_records)
    validation = TubSequence(kl, cfg, validation_records)
    assert len(validation) > 0, "Not enough validation data, decrease the " \
                                "batch size or add more data."

    history = kl.train(model_path=output_path, train_data=training,
                       train_steps=len(training), batch_size=batch_size,
                       validation_data=validation,
                       validation_steps=len(validation),
                       epochs=cfg.MAX_EPOCHS, verbose=cfg.VERBOSE_TRAIN,
                       min_delta=cfg.MIN_DELTA,
                       patience=cfg.EARLY_STOP_PATIENCE)

    return history


