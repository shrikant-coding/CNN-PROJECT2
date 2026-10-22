from pathlib import Path
import json
import tensorflow as tf
from tensorflow.keras import layers, models

# Expected structure:
# animal_classifier_project/
#   train.py
#   raw-img/
#     cane/ cavallo/ elefante/ farfalla/ gallina/
#     gatto/ mucca/ pecora/ ragno/ scoiattolo/

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "raw-img"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "animal_cnn.keras"
CLASS_PATH = MODEL_DIR / "class_names.json"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
SEED = 123
EPOCHS = 12

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATA_DIR}\n"
        "Put the complete raw-img folder next to train.py."
    )

# Load images from the class subfolders and create a reproducible validation split.
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
)

class_names = train_ds.class_names
if len(class_names) < 2:
    raise ValueError("Expected at least two class folders inside raw-img.")

translations = {
    "cane": "Dog",
    "cavallo": "Horse",
    "elefante": "Elephant",
    "farfalla": "Butterfly",
    "gallina": "Chicken",
    "gatto": "Cat",
    "mucca": "Cow",
    "pecora": "Sheep",
    "ragno": "Spider",
    "scoiattolo": "Squirrel",
}
class_info = [
    {"folder": name, "label": translations.get(name, name.replace("_", " ").title())}
    for name in class_names
]
with open(CLASS_PATH, "w", encoding="utf-8") as f:
    json.dump(class_info, f, indent=2, ensure_ascii=False)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# A compact CNN trained from scratch.
model = models.Sequential([
    layers.Input(shape=(*IMAGE_SIZE, 3)),
    layers.Rescaling(1.0 / 255),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.35),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.25),
    layers.Dense(len(class_names), activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor="val_accuracy", save_best_only=True, mode="max"
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    ),
]

print("\nClasses in model output order:")
for i, item in enumerate(class_info):
    print(f"{i}: {item['label']} (folder: {item['folder']})")

model.summary()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# Save the final best weights/model and evaluate on the held-out validation split.
model.save(MODEL_PATH)
loss, accuracy = model.evaluate(val_ds, verbose=1)
print(f"\nValidation accuracy: {accuracy * 100:.2f}%")
print(f"Saved model: {MODEL_PATH}")
print(f"Saved class labels: {CLASS_PATH}")
