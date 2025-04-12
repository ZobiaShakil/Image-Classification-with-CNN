# 1. Import Libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import keras_tuner as kt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib

# 2. Load the Data
# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("Original Training set shape:", x_train.shape)
print("Original Test set shape:", x_test.shape)

# Convert pixel values to float32
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')

# 3. Create tf.data Datasets with Efficient Preprocessing
# Function to resize and preprocess images
def process_image(image, label):
    image = tf.image.resize(image, (224,224))
    image = preprocess_input(image)
    return image, label

batch_size = 32

# Create datasets from tensors with fixed batch sizes (drop_remainder=True ensures fixed shape)
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(buffer_size=5000).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = test_ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

# One-hot encode labels within the dataset
def encode_label(image, label):
    label = tf.one_hot(tf.squeeze(label), depth=10)
    return image, label

train_ds = train_ds.map(encode_label, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(encode_label, num_parallel_calls=tf.data.AUTOTUNE)

# 4. Build a Transfer Learning Model using EfficientNetB0
inputs = Input(shape=(224,224,3))
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=inputs)
base_model.trainable = False
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(10, activation='softmax')(x)

transfer_model = Model(inputs, outputs)

# 5. Compile the Transfer Learning Model
transfer_model.compile(optimizer='adam', 
                       loss='categorical_crossentropy', 
                       metrics=['accuracy'])

# 6. Hyperparameter Tuning with Keras Tuner
def model_builder(hp):
    inp = Input(shape=(224,224,3))
    base = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=inp)
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    dense_units = hp.Int('dense_units', min_value=64, max_value=256, step=64)
    x = Dense(dense_units, activation='relu')(x)
    dropout_rate = hp.Float('dropout_rate', min_value=0.3, max_value=0.7, step=0.1)
    x = Dropout(dropout_rate)(x)
    out = Dense(10, activation='softmax')(x)
    
    model = Model(inp, out)
    learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

tuner = kt.RandomSearch(
    model_builder,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='tuner_dir',
    project_name='churn_transfer'
)

early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

# Compute steps_per_epoch and validation_steps from train_ds and test_ds
# Assuming train_ds and test_ds are finite; you can count batches as:
steps_per_epoch = 100  
validation_steps = 50  
tuner.search(train_ds, epochs=8, steps_per_epoch=steps_per_epoch,
             validation_data=test_ds, validation_steps=validation_steps,
             callbacks=[early_stop])

best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best hyperparameters:", best_hp.values)
best_model = tuner.get_best_models(num_models=1)[0]

# 7. Evaluate the Tuned Model on Test Data
test_loss, test_accuracy = best_model.evaluate(test_ds, verbose=0)
print("Tuned Transfer Learning Model Test Accuracy: {:.2f}%".format(test_accuracy * 100))

y_pred = best_model.predict(test_ds)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = []
for images, labels in test_ds:
    y_true.extend(np.argmax(labels.numpy(), axis=1))

from sklearn.metrics import classification_report, confusion_matrix
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))

# 8. Model Explainability: Grad-CAM
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

last_conv_layer = None
for layer in best_model.layers[::-1]:
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer = layer.name
        break
print("Last Conv Layer:", last_conv_layer)

# Get one batch from the test dataset and select one image
for sample_batch, _ in test_ds.take(1):
    sample_img = sample_batch[0:1]  # shape (1, 224,224,3)
    break

heatmap = make_gradcam_heatmap(sample_img, best_model, last_conv_layer)
plt.matshow(heatmap)
plt.title("Grad-CAM Heatmap")
plt.colorbar()
plt.show()

# 9. Save the Final Tuned Model
best_model.save("final_tuned_transfer_model.h5")
print("Final tuned transfer learning model saved!")

