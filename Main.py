# Import Data Science Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2 as cv

# Tensorflow Libraries
from tensorflow import keras
from tensorflow.keras import layers,models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import Callback, EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Model
from tensorflow.keras.layers import Rescaling, Resizing

# System libraries
from pathlib import Path
import os

# Metrics
from sklearn.metrics import classification_report, confusion_matrix
import itertoolsdataset = "../input/fire-dataset/fire_dataset"
## Check whole format from all images
for root, dirs, files in os.walk(dataset):
    print(root, end="\n")
    #print(root.split('/')[-1], "::+++++++++++++", len(files))
    lol = set()
    if len(files)>0:
        for f in files:
            lol.add(f.split(".")[-1])
print(lol)
image_dir = Path(dataset)

# Get filepaths and labels
filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) + list(image_dir.glob(r'**/*.png'))

#---------------------------------------------------------------------------------
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))
'''labels = []
for x in filepaths:
    #print(os.path.split(os.path.split(x)[0])[1])
    labels.append(os.path.split(os.path.split(x)[0])[1])
    break'''
#---------------------------------------------------------------------------------
# Create DataFrame
image_df = pd.DataFrame(columns=["filepaths", "labels"])
image_df['filepaths'] = filepaths
image_df['labels'] = labels

image_df['filepaths']=image_df['filepaths'].astype(str)
image_df.tail()
import matplotlib.image as mpimg

# Display 16 picture of the dataset with their labels
random_index = np.random.randint(0, len(image_df), 16)
fig, ax = plt.subplots(2, 3, figsize=(10, 5), 
                       subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(ax.flat):
    image = Image.open(image_df.filepaths[random_index[i]])
    ax.imshow(image)
    ax.set_title(image_df.labels[random_index[i]], color='red')
    
plt.tight_layout()
plt.show()
train_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2
)

test_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)
# Split the data into three categories.
train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='filepaths',
    y_col='labels',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='training'
)
val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='filepaths',
    y_col='labels',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='validation'
)
#--------------------------------------------------------------------------
test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='filepaths',
    y_col='labels',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)
pretrained_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None,  # Loaded offline
    pooling='avg'
)

# Load weights from the uploaded file
pretrained_model.load_weights('/kaggle/input/models/fires_classif_model.weights.h5')

# Freeze model layers
for layer in pretrained_model.layers:
    layer.trainable = False
checkpoint_path = "fires_classif_model.weights.h5"
checkpoint_callback = ModelCheckpoint(
    checkpoint_path,
    save_weights_only=True,
    monitor="val_accuracy",  # Because we used 'accuracy' in compile metrics
    save_best_only=True,
    verbose=1
)
model.compile(
    optimizer=Adam(0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=20,
    callbacks=[early_stopping, reduce_lr, checkpoint_callback]
)
def plot_loss_curves(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Plot loss
    plt.figure(figsize =(10,4))
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.figure(figsize =(10,4))
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();

print("    Test Loss: {:.5f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))
# Predict the label of the test_images
pred = model.predict(test_images)
pred = np.argmax(pred,axis=1)

# Map the label
labels = (train_images.class_indices)
labels = dict((v,k) for k,v in labels.items())
pred = [labels[k] for k in pred]

# Display the result
print(f'The first 5 predictions: {pred[:5]}')
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

# === Directory paths ===
fire_dir = "/kaggle/input/fire-dataset/fire_dataset/fire_images"
non_fire_dir = "/kaggle/input/fire-dataset/fire_dataset/non_fire_images"

# === Model input image size ===
img_size = (224, 224)

# === Helper function to load and preprocess images ===
def preprocess_image(img_path, target_size):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0), img

# === Select 5 random fire images and 5 random non-fire images ===
fire_images = np.random.choice(os.listdir(fire_dir), 5, replace=False)
non_fire_images = np.random.choice(os.listdir(non_fire_dir), 5, replace=False)

# === Combine image paths with their true labels ===
all_image_paths = [
    (os.path.join(fire_dir, fname), "fire", 0) for fname in fire_images
] + [
    (os.path.join(non_fire_dir, fname), "non fire", 1) for fname in non_fire_images
]

# Shuffle the images
np.random.shuffle(all_image_paths)

# === Set up for displaying images in 2 rows (5 images per row) ===
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

# Flatten axes to easily iterate over them
axes = axes.flatten()

# === Loop through and predict ===
for i, (img_path, true_label, true_class) in enumerate(all_image_paths):
    try:
        img_array, original_img = preprocess_image(img_path, img_size)
        prediction = model.predict(img_array)
        probability = prediction[0][0]

        # Determine predicted label
        if probability < 0.5:
            predicted_label = "non fire"
        else:
            predicted_label = "fire"

        # Plot each image
        axes[i].imshow(original_img)
        axes[i].set_title(
            f"True: {true_label} | Predicted: {predicted_label} ({probability * 100:.2f}%)",
            fontsize=10
        )
        axes[i].axis('off')

    except Exception as e:
        print(f"Failed to process {img_path}: {e}")

# Adjust layout
plt.tight_layout()
plt.show()

def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(15, 7), text_size=10, norm=False, savefig=False): 
    """Makes a labelled confusion matrix comparing predictions and ground truth labels.

    If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.

  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    norm: normalize values or not (default=False).
    savefig: save confusion matrix to file (default=False).
  
  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.

  Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
    """  
  # Create the confustion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
    n_classes = cm.shape[0] # find the number of classes we're dealing with

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
    fig.colorbar(cax)

    # Are there a list of classes?
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])
  
    # Label the axes
    ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes), # create enough axis slots for each class
         yticks=np.arange(n_classes), 
         xticklabels=labels, # axes will labeled with class names (if they exist) or ints
         yticklabels=labels)
  
    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    ### Added: Rotate xticks for readability & increase font size (required due to such a large confusion matrix)
    plt.xticks(rotation=90, fontsize=text_size)
    plt.yticks(fontsize=text_size)

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
                size=text_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)

  # Save the figure to the current working directory
    if savefig:
        fig.savefig("confusion_matrix.png")
make_confusion_matrix(y_test, pred, list(labels.values()), figsize=(3, 3))
