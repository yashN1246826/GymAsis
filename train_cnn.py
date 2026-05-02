"""
train_cnn.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend (safe on all platforms)
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------------------------------------------ #
#  Configuration                                                      #
# ------------------------------------------------------------------ #

TRAIN_DIR        = 'dataset/train'
TEST_DIR         = 'dataset/test'
MODEL_SAVE_PATH  = 'gym_cnn_model.h5'

IMG_HEIGHT    = 128
IMG_WIDTH     = 128
BATCH_SIZE    = 16       # Small batch — works well with small datasets
EPOCHS_P1     = 20       # Phase 1: head only
EPOCHS_P2     = 10       # Phase 2: fine-tuning
LEARNING_RATE = 0.0001

# MUST match predict_image.py CLASS_NAMES and dataset folder names
CLASS_NAMES = [
    'barbell', 'bench', 'dumbbell', 'kettlebell',
    'pull_up_bar', 'rowing_machine', 'squat_rack', 'treadmill'
]
NUM_CLASSES = len(CLASS_NAMES)

# ------------------------------------------------------------------ #
#  Check dataset exists                                               #
# ------------------------------------------------------------------ #

if not os.path.exists(TRAIN_DIR):
    print(f"ERROR: Training directory '{TRAIN_DIR}' not found.")
    print("Please run 'python setup_dataset.py' first to download and organise images.")
    exit(1)

# ------------------------------------------------------------------ #
#  Data generators                                                    #
# ------------------------------------------------------------------ #

print("[train_cnn] Setting up data generators...")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.20,
    shear_range=0.10,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    classes=CLASS_NAMES
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    classes=CLASS_NAMES
)

print(f"[train_cnn] Training samples : {train_generator.samples}")
print(f"[train_cnn] Test samples     : {test_generator.samples}")
print(f"[train_cnn] Classes          : {train_generator.class_indices}")

# ------------------------------------------------------------------ #
#  Model: MobileNetV2 + custom head                                  #
# ------------------------------------------------------------------ #

print("\n[train_cnn] Building model...")

base_model = MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze for Phase 1

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ------------------------------------------------------------------ #
#  Phase 1: Train head only                                           #
# ------------------------------------------------------------------ #

print("\n[train_cnn] Phase 1: Training classification head (base frozen)...")

history1 = model.fit(
    train_generator,
    epochs=EPOCHS_P1,
    validation_data=test_generator,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=5,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint('gym_cnn_checkpoint.h5', monitor='val_accuracy',
                        save_best_only=True, verbose=0)
    ],
    verbose=1
)

# ------------------------------------------------------------------ #
#  Phase 2: Fine-tune top layers                                      #
# ------------------------------------------------------------------ #

print("\n[train_cnn] Phase 2: Fine-tuning top 30 layers of MobileNetV2...")

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_generator,
    epochs=EPOCHS_P2,
    validation_data=test_generator,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=5,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint('gym_cnn_checkpoint.h5', monitor='val_accuracy',
                        save_best_only=True, verbose=0)
    ],
    verbose=1
)

# ------------------------------------------------------------------ #
#  Evaluation                                                         #
# ------------------------------------------------------------------ #

print("\n[train_cnn] Evaluating on test set...")
test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print(f"\n[train_cnn] Test Accuracy : {test_acc * 100:.2f}%")
print(f"[train_cnn] Test Loss     : {test_loss:.4f}")

test_generator.reset()
preds = model.predict(test_generator, verbose=0)
pred_classes = np.argmax(preds, axis=1)
true_classes = test_generator.classes

print("\n--- Classification Report ---")
print(classification_report(true_classes, pred_classes, target_names=CLASS_NAMES))

print("--- Confusion Matrix ---")
print(confusion_matrix(true_classes, pred_classes))

# ------------------------------------------------------------------ #
#  Save model                                                         #
# ------------------------------------------------------------------ #

model.save(MODEL_SAVE_PATH)
print(f"\n[train_cnn] Model saved: '{MODEL_SAVE_PATH}'")

# ------------------------------------------------------------------ #
#  Plot training history                                              #
# ------------------------------------------------------------------ #

acc  = history1.history['accuracy']  + history2.history['accuracy']
vacc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss  = history1.history['loss']  + history2.history['loss']
vloss = history1.history['val_loss'] + history2.history['val_loss']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(acc,  label='Train Accuracy')
ax1.plot(vacc, label='Val Accuracy')
ax1.set_title('Accuracy over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

ax2.plot(loss,  label='Train Loss')
ax2.plot(vloss, label='Val Loss')
ax2.set_title('Loss over Epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
print("[train_cnn] Training history plot saved: 'training_history.png'")
