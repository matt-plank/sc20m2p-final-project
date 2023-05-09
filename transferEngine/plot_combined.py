import sys

from keras.models import load_model
from keras.utils import CustomObjectScope

from transferEngine.images import image_factory
from transferEngine.models import losses
from transferEngine.models.model import AutoEncoder
from transferEngine.plotting import combined_figure

MODEL_PATH: str = "model.tf"
IMAGE_PATH_1: str = sys.argv[1]
IMAGE_PATH_2: str = sys.argv[2]
ALPHA: float = float(sys.argv[3])
SAVE_PATH: str = sys.argv[4]

with CustomObjectScope({"AutoEncoder": AutoEncoder, "loss": losses.combined_loss(0.5, 0.3, 0.0, 0.1)}):
    model: AutoEncoder = load_model(MODEL_PATH)  # type: ignore

img_1 = image_factory.image_from_path(IMAGE_PATH_1, (64, 64)).wrapped_matrix
img_2 = image_factory.image_from_path(IMAGE_PATH_2, (64, 64)).wrapped_matrix

decoded_1, decoded_2, combined = model.encode_and_combine(img_1, img_2, ALPHA)

fig = combined_figure(img_1[0], img_2[0], combined[0])
fig.savefig(f"model.tf/{SAVE_PATH}")
