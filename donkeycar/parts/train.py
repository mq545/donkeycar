import os
import numpy as np
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import List, Dict, Any, Tuple, Union
from tensorflow.python.keras.utils.data_utils import Sequence as TfSequence
from donkeycar.parts.tub_v2 import Tub
from donkeycar.utils import get_model_by_type, load_image_arr, \
    train_test_split, normalize_image
from donkeycar.parts.keras import KerasPilot
from donkeycar.config import Config
from donkeycar.pipeline.sequence import TubDataset, TubSequence, Pipeline
from donkeycar.pipeline.types import TubRecord

# for typing
Record = Dict[str, Any]
Records = List[Record]
X = TypeVar('X', covariant=True)
Y = TypeVar('Y', covariant=True)


DEFAULT_TRANSFORMATIONS = ['ImageReader', 'ImageNormalizer']


class TubDataset1(object):
    """
    Loads the dataset, and creates a train/test split.
    """

    def __init__(self,
                 tub_paths: List[str],
                 test_size: float = 0.2,
                 shuffle: bool = True) -> None:
        """
        :param tub_paths:   list of paths to tubs
        :param test_size:   test / train split
        :param shuffle:     if shuffling the data or not
        """
        self.tub_paths = tub_paths
        self.test_size = test_size
        self.shuffle = shuffle
        self.tubs = [Tub(tub_path, read_only=True) for tub_path in
                     self.tub_paths]
        self.records: Records = list()

    def train_test_split(self) -> Tuple[Records, Records]:
        print('Loading tubs from paths %s' % self.tub_paths)
        for tub in self.tubs:
            for record in tub:
                record['_image_base_path'] = tub.images_base_path
                self.records.append(record)

        return train_test_split(self.records, shuffle=self.shuffle,
                                test_size=self.test_size)


class RecordTransformer(ABC):
    """ Base class for record transformations which can be stacked on top of
        each other """

    def __init__(self,
                 key: str,
                 config: Config,
                 cache: bool = False) -> None:
        """
        :param key:     key on which the transformation should apply
        :param config:  donkey config
        :param cache:   if transformed record should be put back for
                        performance - this destroys the original record
        """
        self.config = config
        self.key = key
        self.cache = cache

    def get(self, record: Record, key: str, val: Any) -> Any:
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
    def transform(self, record: Record, val: Any) -> Any:
        """ This has to be implemented in derived classes"""
        pass

    @classmethod
    def create(cls,
               transform_names: List[str],
               config: Config) -> List["RecordTransformer"]:
        """ Method to create a stack of transformations on top of the
            LazyRecord """
        transforms = [globals()[name](config) for name in transform_names]
        return transforms


class ImageReader(RecordTransformer):
    """ Convert path into image array """

    def __init__(self, config: Config) -> None:
        super().__init__('cam/image_array', config, True)

    def transform(self, record: Record, val: Any) -> Any:
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

    def __init__(self, config: Config) -> None:
        super().__init__('cam/image_array', config, False)

    def transform(self, record: Record, val: np.ndarray) -> np.ndarray:
        return normalize_image(val)


class LazyRecord(object):
    """ Lazy record which wraps around record dictionary. There is no
        additional functionality here, it's all in the derived classes. """

    def __init__(self,
                 record: Record,
                 model: KerasPilot,
                 transforms: List[RecordTransformer]) -> None:
        self.model = model
        self.record = record
        self.transforms = transforms

    def get_x(self) -> Any:
        """ Dispatch returning of X to the model which will call
            get_entry() on the LazyRecord and from there drive the
            transformations. """
        return self.model.lazy_record_transform_x(self)

    def get_y(self) -> Any:
        """ Dispatch returning of Y to the model which will call
            get_entry() on the LazyRecord and from there drive the
            transformations. """
        return self.model.lazy_record_transform_y(self)

    def get_entry(self, key: str) -> Any:
        """ Get entry from record and run through transformations """
        val = self.record.get(key)
        for transform in self.transforms:
            val = transform.get(self.record, key, val)
        return val


class TubSequence1(Sequence):
    """ Converts sequence of records to lazy records. """
    def __init__(self,
                 model: KerasPilot,
                 config: Config,
                 records: Records,
                 is_train: bool = True) -> None:
        self.model = model
        self.config = config
        self.records = records
        cfg_attr = 'TRAIN_TRANSFORMATIONS' if is_train else \
            'VALIDATION_TRANSFORMATIONS'
        transforms = getattr(self.config, cfg_attr, DEFAULT_TRANSFORMATIONS)
        self.transformations = RecordTransformer.create(transforms, self.config)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> LazyRecord:
        record = self.records[index]
        lazy_record = LazyRecord(record, self.model, self.transformations)
        return lazy_record


class BatchSequence1(TfSequence):
    def __init__(self,
                 lazy_records: TubSequence,
                 batch_size: int) -> None:
        self.lazy_records = lazy_records
        self.batch_size = batch_size

    def __len__(self) -> float:
        return len(self.lazy_records) // self.batch_size

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        count = 0
        x = []
        y = []
        # collecting across the whole batch
        while count < self.batch_size:
            i = (index * self.batch_size) + count
            if i >= len(self.lazy_records):
                break
            this_record = self.lazy_records[i]
            single_x = this_record.get_x()
            single_y = this_record.get_y()
            x.append(single_x)
            y.append(single_y)
            count += 1

        # reshape X, Y
        def reshape(z):
            # each entry in z could either be a single value, or a numpy
            # array, or a tuple containing values and numpy arrays
            if type(z[0]) is tuple:
                dim = len(z[0])
                ret_z = []
                for j in range(dim):
                    z_j = np.array([zi[j] for zi in z])
                    ret_z.append(z_j)
                return ret_z
            else:
                return np.array(z)

        x_res = reshape(x)
        y_res = reshape(y)
        return x_res, y_res








class BatchSequence(TfSequence):
    def __init__(self,
                 pipeline: Iterable[Tuple[X, Y]],
                 batch_size: int) -> None:
        self.pipeline = pipeline
        self.batch_size = batch_size

    def __len__(self) -> int:
        return len(self.pipeline) // self.batch_size

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        count = 0
        x = []
        y = []
        # collecting across the whole batch
        while count < self.batch_size:
            i = (index * self.batch_size) + count
            if i >= len(self.pipeline):
                break
            single_x, single_y = self.pipeline[i]
            x.append(single_x)
            y.append(single_y)
            count += 1

        x_res = np.array(x)
        y_res = np.array(y)
        return x_res, y_res


def make_tf_data(pipeline, batch_size):
    gen = lambda: pipeline
    dataset = tf.data.Dataset.from_generator(
        generator=gen,
        output_types=(tf.float64, tf.float64))
        #output_shapes=(tf.TensorShape([120, 160, 3]), tf.TensorShape([2,])))

    return dataset.repeat().batch(batch_size)


def train(cfg: Config,
          tub_paths: Union[str, List[str]],
          output_path: str,
          model_type: str) -> Dict[str, Any]:
    """
    Train the model
    :param cfg:         donkey config
    :param tub_paths:   single path or list of multiple paths for tubs
    :param output_path: output model path
    :param model_type:  model type, e.g linear, categorical, etc
    :return:            history dictionary
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

    batch_size: int = cfg.BATCH_SIZE
    # loading all records into a single data set
    dataset = TubDataset(tub_paths, config=cfg)
    records = dataset.load_records()
    training_records, validation_records \
        = train_test_split(records, shuffle=False,
                           test_size=(1. - cfg.TRAIN_TEST_SPLIT))
    print('Records # Training %s' % len(training_records))
    print('Records # Validation %s' % len(validation_records))

    # step 1 of pipeline, create the sequence:
    training = TubSequence(records=training_records)
    validation = TubSequence(records=validation_records)

    # step 2 of pipeline, extract X, Y sequence from data
    # get X from tub record:
    def get_X(t: TubRecord) -> np.ndarray:
        img_arr = t.image(cached=True, normalize=True)
        return img_arr

    def get_Y(t: TubRecord) -> np.ndarray:
        y1 = t.underlying['user/angle']
        y2 = t.underlying['user/throttle']
        return np.array([y1, y2])

    # TODO: training_pipe iterates only once and then is exhausted. That's
    #  why keras training fails after one epoch.
    # training_pipe = training.build_pipeline(get_X, get_Y)
    # validation_pipe = validation.build_pipeline(get_X, get_Y)

    # # this version is working.
    training_pipe = Pipeline(training, get_X, get_Y)
    validation_pipe = Pipeline(training, get_X, get_Y)

    # step 3 of pipeline, transform into tf.data or tf.sequence
    # using tf.Data disabled.
    # dataset_train = make_tf_data(training_pipe, cfg.BATCH_SIZE)
    # dataset_validate = make_tf_data(validation_pipe, cfg.BATCH_SIZE)

    dataset_train = BatchSequence(training_pipe, cfg.BATCH_SIZE)
    dataset_validate = BatchSequence(validation_pipe, cfg.BATCH_SIZE)

    assert len(dataset_validate) > 0, \
        "Not enough validation data, decrease the batch size or add more data."

    history = kl.train(model_path=output_path,
                       train_data=dataset_train,
                       train_steps=len(dataset_train),
                       batch_size=batch_size,
                       validation_data=dataset_validate,
                       validation_steps=len(dataset_validate),
                       epochs=cfg.MAX_EPOCHS,
                       verbose=cfg.VERBOSE_TRAIN,
                       min_delta=cfg.MIN_DELTA,
                       patience=cfg.EARLY_STOP_PATIENCE)

    return history
