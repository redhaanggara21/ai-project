import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def train_cnn(dataset_name, update_progress):
    """
    Training CNN untuk image classification.
    Dataset harus berada di folder:
    dataset/{dataset_name}/train/ dan dataset/{dataset_name}/test/
    """
    base_path = os.path.join("dataset", dataset_name)
    train_dir = os.path.join(base_path, "training")
    test_dir = os.path.join(base_path, "validation")

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {base_path}")

    update_progress(10, "📂 Memuat dataset dan preprocessing gambar...")

    # Data generator
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1.0/255)

    train_set = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical'
    )

    test_set = test_datagen.flow_from_directory(
        test_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical'
    )

    update_progress(25, "🧠 Membangun model CNN...")

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(train_set.num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    update_progress(45, "🚀 Training CNN model...")

    # Training model
    history = model.fit(
        train_set,
        validation_data=test_set,
        epochs=5,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: update_progress(
                    45 + int((epoch+1)/5*30),
                    f"Training epoch {epoch+1}/5 - acc: {logs['accuracy']:.2f}"
                )
            )
        ]
    )

    update_progress(80, "📊 Evaluasi model...")

    # Evaluasi model
    loss, accuracy = model.evaluate(test_set, verbose=0)

    update_progress(95, "💾 Menyimpan model dan hasil...")

    # Simpan model
    save_path = os.path.join("dataset", dataset_name, "cnn_model.h5")
    model.save(save_path)

    update_progress(100, "✅ CNN training selesai.")

    return {
        "algorithm": "CNN",
        "dataset": dataset_name,
        "accuracy": float(accuracy),
        "loss": float(loss),
        "model_path": save_path,
        "num_classes": train_set.num_classes,
        "class_indices": train_set.class_indices,
    }
