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
