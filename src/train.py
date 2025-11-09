import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import argparse

def build_datasets(data_dir, batch_size=16, img_size=(224, 224)):
    """Create train, val, test datasets from directories."""
    train_path = os.path.join(data_dir, "train")
    val_path   = os.path.join(data_dir, "val")
    test_path  = os.path.join(data_dir, "test")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_gen = val_test_datagen.flow_from_directory(
        val_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_gen = val_test_datagen.flow_from_directory(
        test_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, val_gen, test_gen

def build_model(num_classes, img_size=(224, 224)):
    """Builds a simple MobileNetV2-based classifier."""
    base_model = MobileNetV2(weights='imagenet', include_top=False,
                             input_shape=img_size + (3,))
    base_model.trainable = False  # Freeze feature extractor

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def main(data_dir, out_path, epochs):
    train_gen, val_gen, test_gen = build_datasets(data_dir)

    num_classes = len(train_gen.class_indices)
    model = build_model(num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Save the best model automatically
    checkpoint = ModelCheckpoint(out_path, monitor='val_accuracy',
                                 save_best_only=True, verbose=1)
    earlystop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[checkpoint, earlystop]
    )

    # Evaluate on test data
    print("\nEvaluating on test data...")
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"✅ Test Accuracy: {test_acc:.4f}")

    # Save final model
    model.save(out_path)
    print(f"\nModel saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_path", type=str, default="models/best_model.keras")
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()
    main(args.data_dir, args.out_path, args.epochs)
