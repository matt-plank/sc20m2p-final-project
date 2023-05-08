import io

import numpy as np
import tensorflow as tf
from flask import Flask, request
from flask_restful import Api, Resource
from keras.models import load_model
from PIL import Image

from transferEngine.models.model import AutoEncoder

MODEL_PATH: str = "model.tf"


class ImageUpload(Resource):
    """A resource for encoding and combining images."""

    def post(self):
        """Receive two images and return the encoded and combined images."""
        image1 = request.files["image1"]
        image2 = request.files["image2"]

        # Convert images to NumPy arrays
        img1 = Image.open(io.BytesIO(image1.read()))
        img2 = Image.open(io.BytesIO(image2.read()))
        arr1 = np.array(img1)
        arr2 = np.array(img2)

        # Resize images to 64x64 with tensorflow
        arr1 = tf.image.resize(arr1, (64, 64)).numpy()  # type: ignore
        arr2 = tf.image.resize(arr2, (64, 64)).numpy()  # type: ignore

        # Load the model
        model: AutoEncoder = load_model(MODEL_PATH)  # type: ignore

        # Encode and combine images
        decoded_1, decoded_2, decoded_combined = model.encode_and_combine(arr1, arr2, 0.5)

        return {
            "image1": Image.fromarray(decoded_1[0]),
            "image2": Image.fromarray(decoded_2[0]),
            "combined": Image.fromarray(decoded_combined[0]),
        }


def create_app():
    app = Flask(__name__)
    api = Api(app)

    api.add_resource(ImageUpload, "/upload")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
