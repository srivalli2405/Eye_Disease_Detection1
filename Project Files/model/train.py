import argparse, os, datetime, tensorflow as tf
from tensorflow.keras import layers, optimizers, callbacks, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def build_model(num_classes):
    base = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    base.trainable = False  # freeze base layers

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(base.input, outputs)
    model.compile(optimizer=optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main(args):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        horizontal_flip=True,
        rotation_range=20,
        zoom_range=0.2
    )

    train_gen = train_datagen.flow_from_directory(
        args.dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        subset='training'
    )

    val_gen = train_datagen.flow_from_directory(
        args.dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        subset='validation'
    )

    num_classes = train_gen.num_classes
    model = build_model(num_classes)

    ckpt_name = f"eye_disease_{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}.h5"
    ckpt_path = os.path.join('model', ckpt_name)

    cb = [
        callbacks.ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    ]

    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=cb
    )

    model.save(os.path.join('model', 'eye_disease_model.h5'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Eye Disease Detection model")
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    main(args)