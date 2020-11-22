import os
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.utils.data_utils import Sequence
from donkeycar.parts.tub_v2 import Tub
from donkeycar.utils import get_model_by_type, load_image_arr, \
    train_test_split, normalize_image


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
        print('Loading tubs from paths %s' % (self.tub_paths))
        for tub in self.tubs:
            for record in tub:
                record['_image_base_path'] = tub.images_base_path
                self.records.append(record)

        return train_test_split(self.records, shuffle=self.shuffle, test_size=self.test_size)


class LazyRecord(object):

    def __init__(self, record, config):
        self._record = record
        self.config = config

    def get_X_Y(self, model):
        return model.get_X_Y(self)

    def get_entry(self, key):
        val = self._record[key]
        if key == 'cam/image_array' and isinstance(val, str):
            image_path = os.path.join(self._record['_image_base_path'], val)
            image = load_image_arr(image_path, self.config)
            norm_img = normalize_image(image)
            self._record[key] = norm_img
            return norm_img
        return val

    def has_entry(self, key):
        return key in self._record


class TubSequence(Sequence):
    def __init__(self, keras_model, config, records):
        self.keras_model = keras_model
        self.config = config
        self.records = records
        self.batch_size = self.config.BATCH_SIZE

    def __len__(self):
        return len(self.records) // self.batch_size

    def __getitem__(self, index):
        count = 0
        records = []

        while count < self.batch_size:
            i = (index * self.batch_size) + count
            if i >= len(self.records):
                break

            record = LazyRecord(self.records[i], self.config)
            records.append(record)
            count += 1

        X = []
        Y = []
        # collecting across the whole batch
        for record in records:
            single_X, single_Y = record.get_X_Y(self.keras_model)
            X.append(single_X)
            Y.append(single_Y)

        # now we have to transpose as keras expects an array of batch size
        # for each entry in X and y
        X = np.array(X)
        Y = np.array(Y)

        # reshape X, Y if required
        def reshape(Z):
            if Z.shape[1] > 1:
                return [Z[:, i] for i in range(Z.shape[1])]
            elif Z.shape[1] == 1:
                return np.squeeze(Z, axis=1)
            else:
                raise ValueError('Cannot process empty record data')

        X = reshape(X)
        Y = reshape(Y)

        return X, Y


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
    dataset = TubDataset(tub_paths, test_size=(1. - cfg.TRAIN_TEST_SPLIT))
    training_records, validation_records = dataset.train_test_split()
    print('Records # Training %s' % len(training_records))
    print('Records # Validation %s' % len(validation_records))

    training = TubSequence(kl, cfg, training_records)
    validation = TubSequence(kl, cfg, validation_records)
    assert len(validation) > 0, "Not enough validation data, decrease the " \
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
        x=training,
        steps_per_epoch=len(training),
        batch_size=batch_size,
        callbacks=callbacks,
        validation_data=validation,
        validation_steps=len(validation),
        epochs=cfg.MAX_EPOCHS,
        verbose=cfg.VERBOSE_TRAIN,
        workers=1,
        use_multiprocessing=False
    )
    return history


