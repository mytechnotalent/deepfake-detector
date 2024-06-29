![image](https://github.com/mytechnotalent/deepfake-detector/blob/main/deepfake-detector.png?raw=true)

## FREE Reverse Engineering Self-Study Course [HERE](https://github.com/mytechnotalent/Reverse-Engineering-Tutorial)

<br>

# Deepfake Detector
Deepfake Detector is an AI/ML model designed to detect AI-generated or manipulated images.

## CURRENTLY UNDER DEVELOPMENT

## Train
```python
import os
import zipfile
import urllib3
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
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

    def _build_model(self) -> keras.Sequential:
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
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    def compile_model(self):
        """
        Compiles the model with specified optimizer, loss, and metrics.
        """
        self.model.compile(optimizer=keras.optimizers.Adam(
            learning_rate=0.0001), 
            loss=tf.keras.losses.BinaryCrossentropy(), 
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

    def train_model(self, train_data, val_data, epochs=100, batch_size=32):
        """
        Trains the model with the given data.

        Parameters:
        -----------
        train_data : tf.data.Dataset
            The training data.
        val_data : tf.data.Dataset
            The validation data.
        epochs : int, optional
            Number of epochs to train the model (default is 100).
        batch_size : int, optional
            Batch size for training (default is 32).

        Returns:
        --------
        keras.callbacks.History
            The training history.
        """
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        return self.model.fit(
            train_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data,
            callbacks=[early_stopping_cb, reduce_lr_cb]
        )

    def evaluate_model(self, test_data):
        """
        Evaluates the model with the given test data.

        Parameters:
        -----------
        test_data : tf.data.Dataset
            The test data.

        Returns:
        --------
        list
            Evaluation metrics.
        """
        return self.model.evaluate(test_data)

    def save_model(self, file_path):
        """
        Saves the model to the specified file path.

        Parameters:
        -----------
        file_path : str
            The path to save the model.
        """
        self.model.save(file_path)


class ModelTrainer:
    """
    A class to train, save, and evaluate a deepfake detection model.

    Methods:
    --------
    train_save_evaluate_model(train_data, val_data, test_data):
        Trains, saves, and evaluates the deepfake detection model.
    """

    def train_save_evaluate_model(self, train_data, val_data, test_data):
        """
        Trains, saves, and evaluates the deepfake detection model.

        Args:
        - train_data (tf.data.Dataset): Training dataset.
        - val_data (tf.data.Dataset): Validation dataset.
        - test_data (tf.data.Dataset): Test dataset.
        """
        detector = DeepfakeDetectorModel()
        detector.compile_model()
        history = detector.train_model(train_data, val_data)
        detector.save_model('deepfake_detector_model.keras')
        detector.evaluate_model(test_data)
        # plot training history
        pd.DataFrame(history.history).plot(figsize=(8, 5))
        plt.grid(True)
        plt.show()


def main():
    # configuration
    dataset_url = 'https://www.kaggle.com/api/v1/datasets/download/manjilkarki/deepfake-and-real-images?datasetVersionNumber=1'
    dataset_download_dir = 'data'
    dataset_dir = 'data/Dataset'
    dataset_file = 'archive.zip'
    
    # check if TensorFlow can access the GPU
    print('num GPUs available:', len(tf.config.list_physical_devices('GPU')))

    # download, extract dataset, train, evaluate and save model
    dataset_manager = DatasetManager(dataset_url, dataset_download_dir, dataset_file)
    if dataset_manager.download_and_extract_dataset():
        # load and split data
        data_processor = DataProcessor(dataset_dir)
        train_data, test_data, val_data = data_processor.load_split_data()
        
        # augment the training data
        train_data = DataProcessor.get_augmented_data(train_data)
        
        # train, save, and evaluate model
        model_trainer = ModelTrainer()
        model_trainer.train_save_evaluate_model(train_data, val_data, test_data)
    else:
        print('dataset download and extraction failed')


if __name__ == '__main__':
    main()
```

## Inference
```python
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
import numpy as np
import os


def allowed_file(filename):
    """
    Check if a file has an allowed extension.

    Parameters:
    -----------
    filename : str
        The name of the file to check.

    Returns:
    --------
    bool
        True if the file has an allowed extension, False otherwise.
    """
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(file_path):
    """
    Predict whether an image is Real or Fake using the loaded model.

    Parameters:
    -----------
    file_path : str
        The path to the image file.

    Returns:
    --------
    tuple
        A tuple containing the prediction and the prediction percentage.
    """
    img = image.load_img(file_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    result = model.predict(img_array)
    prediction = result[0][0]
    prediction_percentage = prediction * 100
    return prediction, prediction_percentage


# instantiate app w/ config
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
model_path = 'deepfake_detector_model.keras'


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """
    Handle file upload and prediction requests.

    Returns:
    --------
    str
        The rendered HTML template with the result or error message.
    """
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error='no file part')
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', error='no selected file')
        if file and allowed_file(file.filename):
            # save the uploaded file to the uploads directory
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            # predict if the image is Real or Fake
            prediction, prediction_percentage = predict_image(filename)
            # clean up the uploaded file
            os.remove(filename)
            # determine result message
            result = 'Fake' if prediction >= 0.5 else 'Real'
            # render result to the user
            return render_template('index.html', result=result, prediction_percentage=prediction_percentage)
        else:
            return render_template('index.html', error='allowed file types are png, jpg, jpeg')
    return render_template('index.html')


if __name__ == '__main__':
    # load the trained model
    model = load_model(model_path)

    # run model app
    app.run(debug=True)
```

## Step 1a: Setup (MAC)
```bash
brew install python@3.11
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Step 1b: Setup (Ubuntu)
```bash
sudo apt install python3.11
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Step 1c: Setup (Windows)
```bash
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe" -OutFile "python-3.11.9-amd64.exe"
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
python-3.11.9-amd64.exe
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
```

## Step 2: Train (optional)
```bash
python train.py
```

## Step 3: Inference
```
python inference.py
```

## Dataset Reference
Kaggle Dataset [HERE](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)

## Dataset Citation
```
@ Inproceedings{ltnghia-ICCV2021,
  Title          = {OpenForensics: Large-Scale Challenging Dataset For 
Multi-Face Forgery Detection And Segmentation In-The-Wild},
  Author         = {Trung-Nghia Le and Huy H. Nguyen and Junichi Yamagishi 
and Isao Echizen},
  BookTitle      = {International Conference on Computer Vision},
  Year           = {2021}, 
}
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)
