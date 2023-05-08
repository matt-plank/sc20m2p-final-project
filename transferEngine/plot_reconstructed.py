import sys

from keras.models import load_model
from keras.utils import CustomObjectScope

from transferEngine.images import image_factory
from transferEngine.models import losses
from transferEngine.models.model import AutoEncoder
from transferEngine.plotting import image_decode_figure, image_results_figure

MODEL_PATH: str = "model.tf"
IMAGE_PATH: str = sys.argv[1]
SAVE_PATH: str = sys.argv[2]

with CustomObjectScope({"AutoEncoder": AutoEncoder, "loss": losses.combined_loss(0.5, 0.3, 0.0, 0.1)}):
    model: AutoEncoder = load_model(MODEL_PATH)  # type: ignore

img_1 = image_factory.image_from_path(IMAGE_PATH, (64, 64)).wrapped_matrix
decoded_1 = model.predict(img_1.reshape(1, 64, 64, 3))

fig = image_decode_figure(img_1[0], decoded_1[0])
fig.savefig(f"model.tf/{SAVE_PATH}")
