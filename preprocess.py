import tensorflow as tf

def load_data():

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train = datagen.flow_from_directory(
        "data/train",
        target_size=(256,256),
        color_mode="grayscale",
        class_mode="binary",
        subset="training"
    )

    val = datagen.flow_from_directory(
        "data/train",
        target_size=(256,256),
        color_mode="grayscale",
        class_mode="binary",
        subset="validation"
    )

    return train, val