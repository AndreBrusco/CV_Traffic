# pip3 install -r requirements.txt
import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 20
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.1

# Main Function


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")

# Data Loader


def load_data(data_dir):

    images = []
    labels = []

    # Loop through each category directory
    for category in range(NUM_CATEGORIES):
        category_dir = os.path.join(data_dir, str(category))
        if not os.path.isdir(category_dir):
            print(f"Warning: Directory {category_dir} not found. Skipping category {category}.")
            continue

        # Loop through each image in the category directory
        for filename in os.listdir(category_dir):
            filepath = os.path.join(category_dir, filename)
            try:
                # Read image using OpenCV
                image = cv2.imread(filepath)
                if image is None:
                    print(f"Warning: Unable to read image {filepath}. Skipping.")
                    continue

                # Resize image to the desired dimensions
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                # Append image and label to the lists
                images.append(image)
                labels.append(category)
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

    print(f"Loaded {len(images)} images and {len(labels)} labels.")
    return images, labels

# NN Model


def get_model():

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu",
                               input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Compile the model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
