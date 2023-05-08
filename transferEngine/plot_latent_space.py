import sys

import numpy as np
from keras.models import load_model
from keras.utils import CustomObjectScope

from transferEngine.images import image_factory
from transferEngine.models import losses
from transferEngine.models.model import AutoEncoder
from transferEngine.plotting import latent_space_figure

MODEL_PATH: str = "model.tf"
IMAGE_PATH: str = sys.argv[1]
SAVE_PATH: str = sys.argv[2]

with CustomObjectScope({"AutoEncoder": AutoEncoder, "loss": losses.combined_loss(0.5, 0.3, 0.0, 0.1)}):
    model: AutoEncoder = load_model(MODEL_PATH)  # type: ignore

img = image_factory.image_from_path(IMAGE_PATH, (64, 64)).wrapped_matrix
encoded = model.encoder.predict(img.reshape(1, 64, 64, 3))
encoded = np.reshape(encoded, (12, 8, 8))

fig = latent_space_figure(encoded)
fig.savefig(f"model.tf/{SAVE_PATH}")
print(SAVE_PATH)
