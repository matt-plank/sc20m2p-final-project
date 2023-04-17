import logging
from pathlib import Path
from typing import Dict

from keras.callbacks import EarlyStopping

import transferEngine.plotting as plotting
from transferEngine.config import config_from_cli_or_yaml
from transferEngine.images import image_dataset_factory, image_factory
from transferEngine.models import model_factory


def main():
    """The entry point of the program.

    This function will load the images, train the model, and plot the results.
    """
    # Set up logging & create a new logger
    logging.basicConfig(
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger("Main")

    # Get configuration
    logger.info("Loading configuration...")
    config: Dict = config_from_cli_or_yaml()

    # Check if the dataset already exists
    dataset_path = Path(config["dataset_save_path"])
    if dataset_path.exists():
        logger.info("Dataset already exists, loading...")
        image_dataset = image_dataset_factory.dataset_from_pickle(config["dataset_save_path"])
    else:
        logger.info("Dataset does not exist, creating...")
        image_dataset = image_dataset_factory.dataset_from_path(
            config["dataset_path"],
            target_size=config["target_shape"][:2],
            verbose=True,
        )
        image_dataset.save_to_pickle(config["dataset_save_path"])

    # Train the model
    logger.info("Training model...")
    image_matrix = image_dataset.images_as_matrix(augment=True)
    logger.info(f"X shape: {image_matrix.shape}")

    early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=1)

    model, training_history = model_factory.create_and_train_model(
        config["target_shape"],
        image_matrix,
        config["split"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=[early_stopping],
    )

    # Demonstrate on some example images
    logger.info("Demonstrating on example images...")
    img_1 = image_factory.image_from_path("exampleImages/testImage.jpeg", config["target_shape"][:2]).wrapped_matrix
    img_2 = image_factory.image_from_path("exampleImages/testCombine.jpg", config["target_shape"][:2]).wrapped_matrix

    decoded_1, decoded_2, decoded_combined = model.encode_and_combine(img_1, img_2, 0.5)

    # Create and save plots of the results
    logger.info("Saving results...")
    plotting.image_results_figure(img_1[0], img_2[0], decoded_1[0], decoded_2[0], decoded_combined[0]).savefig(f"{config['model_path']}/results.png")
    plotting.training_history_figure(training_history).savefig(f"{config['model_path']}/accuracy.png")


if __name__ == "__main__":
    main()
