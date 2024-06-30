import os
import zipfile
import urllib3
import requests
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models  # type: ignore


class DataLoader:
    """
    A class to handle dataset downloading, unzipping, and loading from directories.

    Attributes:
    -----------
    path : str
        Base path for the dataset.
    """

    def __init__(self, path):
        """
        Initializes DataLoader with the given path.

        Parameters:
        -----------
        path : str
            Base path for the dataset.
        """
        self.path = path

    @staticmethod
    def download_dataset(url, download_dir, file_name):
        """
        Downloads a dataset from the specified URL.

        Parameters:
        -----------
        url : str
            The URL to download the dataset from.
        download_dir : str
            The directory to download the dataset to.
        file_name : str
            The name of the file to save the dataset as.

        Returns:
        --------
        bool
            A flag to indicate download and parse.
        """
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        file_path = os.path.join(download_dir, file_name)
        if not os.path.exists(file_path):
            # disable InsecureRequestWarning
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            response = requests.get(url, stream=True, verify=False)
            total_size = int(response.headers.get('content-length', 0))
            with open(file_path, 'wb') as file, tqdm(
                    desc=file_name,
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
            print(f'dataset downloaded and saved to {file_path}')
            return True
        else:
            print(f'dataset already exists at {file_path}')
            return False

    @staticmethod
    def unzip_dataset(zip_path, extract_to):
        """
        Unzips the dataset.

        Parameters:
        -----------
        zip_path : str
            The path to the zip file.
        extract_to : str
            The directory to extract the contents to.
        """
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

    def get_image_dataset_from_directory(self, dir_name):
        """
        Loads images from the specified directory.

        Parameters:
        -----------
        dir_name : str
            The name of the directory to load images from.

        Returns:
        --------
        tf.data.Dataset
            A dataset of images.
        """
        dir_path = os.path.join(self.path, dir_name)
        return tf.keras.utils.image_dataset_from_directory(
            dir_path,
            labels='inferred',
            color_mode='rgb',
            seed=42,
            batch_size=32,
            image_size=(128, 128)
        )


class DataProcessor:
    """
    A class to process and augment image datasets.

    Attributes:
    -----------
    dataset_dir : str
        Directory path where the dataset is located.

    Methods:
    --------
    load_split_data():
        Loads and splits the dataset into training, testing, and validation sets.

    get_augmented_data(train_data):
        Applies data augmentation to the training data.
    """

    def __init__(self, dataset_dir):
        """
        Initializes the DataProcessor with the dataset directory path.

        Parameters:
        -----------
        dataset_dir : str
            Directory path where the dataset is located.
        """
        self.dataset_dir = dataset_dir

    def load_split_data(self):
        """
        Loads and splits the dataset into training, testing, and validation sets.

        Returns:
        --------
        tuple:
            Three tuples containing the image datasets for training, testing, and validation.
        """
        data_loader = DataLoader(self.dataset_dir)
        train_data = data_loader.get_image_dataset_from_directory('Train')
        test_data = data_loader.get_image_dataset_from_directory('Test')
        val_data = data_loader.get_image_dataset_from_directory('Validation')
        return train_data, test_data, val_data

    @staticmethod
    def get_augmented_data(train_data):
        """
        Applies data augmentation to the training data.

        Parameters:
        -----------
        train_data : tf.data.Dataset
            The training data.

        Returns:
        --------
        tf.data.Dataset
            The augmented training data.
        """
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip('horizontal_and_vertical'),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.2),
        ])
        augmented_data = train_data.map(lambda x, y: (data_augmentation(x, training=True), y), 
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return augmented_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


class DatasetManager:
    """
    A class to manage dataset downloading and extraction.

    Attributes:
    -----------
    dataset_url : str
        URL of the dataset to be downloaded.
    dataset_download_dir : str
        Directory where the dataset will be downloaded and extracted.
    dataset_file : str
        Name of the dataset zip file.

    Methods:
    --------
    download_and_extract_dataset():
        Downloads the dataset from the specified URL and extracts it if not already extracted.
    """

    def __init__(self, dataset_url, dataset_download_dir, dataset_file):
        """
        Initializes the DatasetManager with dataset URL, download directory, and dataset file name.

        Parameters:
        -----------
        dataset_url : str
            URL of the dataset to be downloaded.
        dataset_download_dir : str
            Directory where the dataset will be downloaded and extracted.
        dataset_file : str
            Name of the dataset zip file.
        """
        self.dataset_url = dataset_url
        self.dataset_download_dir = dataset_download_dir
        self.dataset_file = dataset_file

    def download_and_extract_dataset(self):
        """
        Downloads the dataset from the specified URL and extracts it if not already extracted.

        Returns:
        --------
        bool:
            True if download and extraction were successful, False otherwise.
        """
        file_path = os.path.join(self.dataset_download_dir, self.dataset_file)
        # check if the dataset is already downloaded and extracted
        extracted_dir = os.path.join(self.dataset_download_dir, 'Dataset')
        if os.path.exists(extracted_dir):
            print(f'dataset is already downloaded and extracted at {extracted_dir}')
            return True
        # if Dataset directory does not exist, check if the dataset zip file exists
        if not os.path.exists(file_path):
            # download dataset
            download_successful = DataLoader.download_dataset(self.dataset_url, self.dataset_download_dir, self.dataset_file)
            if not download_successful:
                print(f'failed to download dataset from {self.dataset_url}')
                return False
        # extract dataset if zip file exists
        if os.path.exists(file_path):
            DataLoader.unzip_dataset(file_path, self.dataset_download_dir)
            return True
        # finally, if there is something wrong with the path or unable to write to disk
        print(f'dataset file {file_path} not found after download')
        return False


class DeepfakeDetectorModel:
    """
    A class to create and train a deepfake detection model.

    Attributes:
    -----------
    model : keras.Sequential
        The deepfake detection model.
    """

    def __init__(self):
        """
        Initializes the DeepfakeDetectorModel.
        """
        self.model = self._build_model()

    def _build_model(self):
        """
        Builds the deepfake detection model.

        Returns:
        --------
        keras.Sequential
            The deepfake detection model.
        """
        model = models.Sequential()
        model.add(layers.Input(shape=(128, 128, 3)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    def compile_model(self, learning_rate):
        """
        Compiles the model with the specified learning rate.

        Parameters:
        -----------
        learning_rate : float
            The learning rate to use for training.
        """
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def train_model(self, train_data, val_data, epochs):
        """
        Trains the model with the specified data and epochs.

        Parameters:
        -----------
        train_data : tf.data.Dataset
            The training data.
        val_data : tf.data.Dataset
            The validation data.
        epochs : int
            The number of epochs to train for.

        Returns:
        --------
        keras.callbacks.History
            The training history.
        """
        return self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs
        )

    def evaluate_model(self, test_data):
        """
        Evaluates the model with the test data.

        Parameters:
        -----------
        test_data : tf.data.Dataset
            The test data.

        Returns:
        --------
        tuple:
            The loss and accuracy of the model on the test data.
        """
        return self.model.evaluate(test_data)

    def save_model(self, path):
        """
        Saves the trained model to the specified path.

        Parameters:
        -----------
        path : str
            The path to save the model to.
        """
        self.model.save(path)


if __name__ == '__main__':
    # define  dataset URL and file names
    dataset_url = 'https://www.kaggle.com/api/v1/datasets/download/manjilkarki/deepfake-and-real-images?datasetVersionNumber=1'
    dataset_download_dir = './data'
    dataset_file = 'archive.zip'

    # create a DatasetManager instance and download/extract the dataset
    dataset_manager = DatasetManager(dataset_url, dataset_download_dir, dataset_file)
    dataset_manager.download_and_extract_dataset()

    # load and process the dataset
    dataset_dir = './data/Dataset'
    data_processor = DataProcessor(dataset_dir)
    train_data, test_data, val_data = data_processor.load_split_data()

    # apply data augmentation to the training data
    augmented_train_data = DataProcessor.get_augmented_data(train_data)

    # initialize and build the deepfake detection model
    deep_fake_detector_model = DeepfakeDetectorModel()

    # compile the model with a specified learning rate
    learning_rate = 0.0001
    deep_fake_detector_model.compile_model(learning_rate)

    # train the model with the training data and validation data for a specified number of epochs
    epochs = 50
    history = deep_fake_detector_model.train_model(augmented_train_data, val_data, epochs)

    # evaluate the model with the test data
    loss, accuracy = deep_fake_detector_model.evaluate_model(test_data)
    print(f'model test accuracy: {accuracy * 100:.2f}%')

    # save the trained model
    model_save_path = 'deepfake_detector_model'
    deep_fake_detector_model.save_model(model_save_path)
