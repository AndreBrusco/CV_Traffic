## Document ReadME:

# Fase 1:

On this first try this was our choices.
We choose:

    model = tf.keras.Sequential([
        # Convolutional layer with 32 filters, 3x3 kernel, ReLU activation
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Second convolutional layer with 64 filters
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Flatten the convolutional output
        tf.keras.layers.Flatten(),

    # Add a dense hidden layer with 64 neurons
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.1),  # Dropout for regularization

    # Output layer with NUM_CATEGORIES neurons (softmax activation)
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

First Attempt
Parameters:
EPOCHS: 15
IMG_WIDTH, IMG_HEIGHT: 30x30
NUM_CATEGORIES: 43
TEST_SIZE: 0.2
Model Structure:
2 convolutional layers with 32 and 64 filters
Flatten layer, Dense(64), Dropout(0.1)
Output layer with softmax
Results:
Accuracy: 0.0529 (5.29%)
Loss: 3.4886

Results:
From the results, we can see that our CNN network struggled to learn meaningful patterns from the data, achieving an accuracy of only 5.29% and a high loss of 3.4886. This suggests that the model is either underfitting or not complex enough to capture the features required for accurate classification. Factors such as insufficient training epochs, overly aggressive dropout, or suboptimal architecture may have limited its performance.

# Fase 2:

On the second atempt, we choose this follow parameters and structure:
EPOCHS = 20
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.1

    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
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

Second Attempt
Parameters:
EPOCHS: 20
IMG_WIDTH, IMG_HEIGHT: 30x30
NUM_CATEGORIES: 43
TEST_SIZE: 0.1
Model Structure:
3 convolutional layers with 32, 64, and 128 filters
Flatten layer, Dense(128), Dropout(0.5)
Output layer with softmax
Results:
Accuracy: 0.9839 (98.39%)
Loss: 0.0911

# Compare with the old results:

1. Increased Accuracy
   The accuracy improved from 5.29% to 98.39%, demonstrating that the new architecture captured features much better and generalized effectively on the test data.
2. Reduced Loss
   Loss decreased significantly from 3.4886 to 0.0911, indicating the model is now making much better predictions with minimal errors.
3. Enhanced Network Complexity
   Adding a third convolutional layer (128 filters) allowed the model to extract deeper features from the images, critical for distinguishing between 43 categories.
   Using a larger dense layer (128 neurons) likely helped in combining these features effectively for classification.
4. Increased Regularization
   Using a higher dropout rate of 0.5 (vs. 0.1 in the first attempt) helped prevent overfitting, contributing to better generalization.
5. More Epochs
   Training for 20 epochs instead of 15 gave the model more time to learn, which, combined with a lower TEST_SIZE (10% vs. 20%), ensured more data was available for training.

# Professor's Results

Test Accuracy: 0.9535 (95.35%)
Test Loss: 0.1616
Key Features:
Lower accuracy compared to your model.
Higher loss (0.1616) than your loss (0.0911), indicating your model is making fewer errors on the test set.
Comparison
Accuracy:

Your model achieved an accuracy of 98.39%, significantly higher than the professor's 95.35%. This demonstrates that your network architecture and hyperparameter choices better capture the features of the traffic sign dataset.
Loss:

Your loss (0.0911) is much lower than the professor's (0.1616), showing that your predictions are more confident and closer to the ground truth.
Model Complexity:

Your model used an additional convolutional layer with 128 filters, which likely helped in extracting deeper hierarchical features from the images.
You increased dropout regularization to 0.5, reducing overfitting.
Epochs:

You trained for 20 epochs, compared to the professor's 10 epochs. This allowed your model to learn more from the data, contributing to better performance.
