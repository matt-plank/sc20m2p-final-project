from ..images.image_dataset import ImageDataset
from .model import AutoEncoder


def train_model(dataset: ImageDataset, split: float, epochs: int, batch_size: int) -> tuple:
    """Train a new model and return it.

    Args:
        dataset: The dataset to train the model on.
        split: The validation split to use when training the model.
        epochs: The number of epochs to train the model for.
        batch_size: The batch size to use when training the model.
    """
    model: AutoEncoder = AutoEncoder((128, 128, 3))
    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

    image_matrix = dataset.images_as_matrix(augment=True)

    history = model.fit(
        image_matrix,
        image_matrix,
        validation_split=split,
        epochs=epochs,
        batch_size=batch_size,
    )

    return model, history


def train_or_load_model(path: str, dataset: ImageDataset, split: float, epochs: int, batch_size: int) -> tuple:
    """If a model exists at the given path, load it. Otherwise, train a new model and save it to the path.

    Args:
        path: The path to the model.
        dataset: The dataset to train the model on.
    """
    history = None

    try:
        model: AutoEncoder = load_model(path)  # type: ignore
    except OSError:
        model, history = train_model(dataset, split, epochs, batch_size)

    return model, history
