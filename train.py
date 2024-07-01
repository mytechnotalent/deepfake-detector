import os
import zipfile
import urllib3
import requests
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

class DatasetHandler:
    """A class to handle dataset downloading, unzipping, loading, processing, and augmentation."""

    def __init__(self, dataset_url, dataset_download_dir, dataset_file, dataset_dir='./data/Dataset'):
        self.dataset_url = dataset_url
        self.dataset_download_dir = dataset_download_dir
        self.dataset_file = dataset_file
        self.dataset_dir = dataset_dir

    def download_dataset(self):
        if not os.path.exists(self.dataset_download_dir):
            os.makedirs(self.dataset_download_dir)
        file_path = os.path.join(self.dataset_download_dir, self.dataset_file)
        if os.path.exists(file_path):
            print(f'Dataset file {self.dataset_file} already exists at {file_path}')
            return True
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        response = requests.get(self.dataset_url, stream=True, verify=False)
        total_size = int(response.headers.get('content-length', 0))
        with open(file_path, 'wb') as file, tqdm(desc=self.dataset_file, total=total_size, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
        print(f'Dataset downloaded and saved to {file_path}')
        return True

    def unzip_dataset(self):
        file_path = os.path.join(self.dataset_download_dir, self.dataset_file)
        extracted_dir = os.path.join(self.dataset_download_dir, 'Dataset')
        if os.path.exists(extracted_dir):
            print(f'Dataset is already downloaded and extracted at {extracted_dir}')
            return True
        if not os.path.exists(file_path):
            print(f'Dataset file {file_path} not found after download')
            return False
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(self.dataset_download_dir)
        print(f'Dataset extracted to {extracted_dir}')
        return True

    def get_image_dataset_from_directory(self, dir_name):
        dir_path = os.path.join(self.dataset_dir, dir_name)
        return tf.keras.utils.image_dataset_from_directory(
            dir_path,
            labels='inferred',
            color_mode='rgb',
            seed=42,
            batch_size=32,
            image_size=(256, 256)
        )

    def get_augmented_data(self, train_data):
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip('horizontal_and_vertical'),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.2),
            layers.RandomBrightness(0.2),
        ])
        augmented_data = train_data.map(lambda x, y: (data_augmentation(x, training=True), y),
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return augmented_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    def load_split_data(self):
        train_data = self.get_image_dataset_from_directory('Train')
        test_data = self.get_image_dataset_from_directory('Test')
        val_data = self.get_image_dataset_from_directory('Validation')
        return train_data, test_data, val_data


class DeepfakeDetectorModel:
    """A class to create and train a deepfake detection model."""

    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Input(shape=(256, 256, 3)))
        model.add(layers.Rescaling(1./255, name='rescaling'))
        # block 1
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001)))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
        # block 2
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001)))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
        # block 3
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001)))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
        # global average pooling
        model.add(layers.GlobalAveragePooling2D())
        # fully connected layers
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001)))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    def compile_model(self, learning_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        )

    def train_model(self, train_data, val_data, epochs):
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1)
        model_checkpoint_callback = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
        return self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=[early_stopping_callback, reduce_lr_callback, model_checkpoint_callback]
        )

    def evaluate_model(self, test_data):
        return self.model.evaluate(test_data)

    def save_model(self, path):
        self.model.save(path)


class TrainModel:
    """A class to manage training of a deepfake detection model."""

    def __init__(self, dataset_url, dataset_download_dir, dataset_file, dataset_dir='./data/Dataset'):
        self.dataset_handler = DatasetHandler(dataset_url, dataset_download_dir, dataset_file, dataset_dir)

    def run_training(self, learning_rate=0.0001, epochs=50):
        if not self.dataset_handler.download_dataset():
            print('Failed to download dataset.')
            return
        if not self.dataset_handler.unzip_dataset():
            print('Failed to unzip dataset.')
            return
        train_data, test_data, val_data = self.dataset_handler.load_split_data()
        train_data = self.dataset_handler.get_augmented_data(train_data)
        model = DeepfakeDetectorModel()
        model.compile_model(learning_rate)
        history = model.train_model(train_data, val_data, epochs)
        evaluation_metrics = model.evaluate_model(test_data)
        model.save_model('deepfake_detector_model.keras')
        return history, evaluation_metrics


def main():
    dataset_url = 'https://www.kaggle.com/api/v1/datasets/download/manjilkarki/deepfake-and-real-images?datasetVersionNumber=1'
    dataset_download_dir = './data'
    dataset_file = 'dataset.zip'
    train_model_instance = TrainModel(dataset_url, dataset_download_dir, dataset_file)
    history, evaluation_metrics = train_model_instance.run_training(learning_rate=0.0001, epochs=50)
    print('Training complete. Evaluation metrics:', evaluation_metrics)


if __name__ == '__main__':
    main()
