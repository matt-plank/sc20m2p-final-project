import logging
from itertools import cycle
from typing import Dict, List

import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.losses import MSE
from keras.optimizers import Adam

import transferEngine.plotting as plotting
from transferEngine.config import config_from_cli_or_yaml
from transferEngine.data import batch_generator
from transferEngine.images import image_factory
from transferEngine.models import losses, model_factory


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

    # Create a generator for the dataset
    dataset_path: str = config["dataset_path"]
    image_paths: List[str] = batch_generator.images_from_dir(dataset_path)
    training_image_paths, validation_image_paths = batch_generator.split_images(image_paths, config["split"])
    training_dataset = cycle(batch_generator.from_paths(training_image_paths, config["batch_size"]))
    validation_dataset = cycle(batch_generator.from_paths(validation_image_paths, config["batch_size"]))

    # Create the model
    optimizer = Adam(learning_rate=1e-3)
    model = model_factory.create_model(
        config["target_shape"],
        optimizer=optimizer,
        loss=losses.combined_loss(0.5, 0.3, 0.1, 0.1),
    )

    # Train the model
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=12,
        verbose=1,
    )

    reduce_ldr_on_plateau = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=5,
        verbose=1,
        min_lr=5e-6,  # type: ignore - I don't know why VS Code thinks min_lr should be an int - it isn't specified
    )

    training_history = model.fit(
        training_dataset,
        # validation_data=validation_dataset,
        steps_per_epoch=len(image_paths) // config["batch_size"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=[reduce_ldr_on_plateau, early_stopping],
    )

    # Save model in ".tf" format
    logger.info("Saving model...")
    model.save(config["model_path"], save_format="tf")

    # Demonstrate on some example images
    logger.info("Demonstrating on example images...")
    img_1 = image_factory.image_from_path("exampleImages/testImage.jpeg", config["target_shape"][:2]).wrapped_matrix
    img_2 = image_factory.image_from_path("exampleImages/testCombine.jpg", config["target_shape"][:2]).wrapped_matrix
    img_3 = image_factory.image_from_path("exampleImages/Me.jpg", config["target_shape"][:2]).wrapped_matrix

    decoded_1, decoded_2, decoded_combined = model.encode_and_combine(img_1, img_2, config["alpha"])
    decoded_3: np.ndarray = model(img_3).numpy()  # type: ignore

    # Create and save plots of the results
    logger.info("Saving results...")
    plotting.image_results_figure(img_1[0], img_2[0], img_3[0], decoded_1[0], decoded_2[0], decoded_3[0], decoded_combined[0]).savefig(f"{config['model_path']}/results.png")
    plotting.training_history_figure(training_history).savefig(f"{config['model_path']}/accuracy.png")


if __name__ == "__main__":
    main()
