import tensorflow as tf
import keras
from keras import layers
import os


def parse_image(img):

    decoded = tf.image.decode_jpeg(img)
    resized = tf.image.resize_images(decoded, [28, 28])
    greyscale = tf.image.rgb_to_grayscale(resized)

    # TO SEE GREYSCALE VALUES UNCOMMENT
    # with tf.Session() as sess:
    #
    #     greyscale_img = sess.run(greyscale)
    #     print(greyscale_img)

    return greyscale


def iterate_through_directories(rootdir):
    directory = os.fsencode(rootdir)

    # list = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]

    # for subdir in os.listdir(directory):
    #     if os.path.isdir(os.path.join(directory, subdir)):
    #         basename = os.path.basename(subdir)
    #         for file in subdir:
    #             print(basename, file)

    images = []
    labels = []
    class_names = []

    for path, dirs, files in os.walk(directory):

        base_name = os.path.basename(path)

        class_names.append(base_name)

        for file in files:

            img = tf.read_file(file)
            images.append(img)
            labels.append(base_name)

    return images, labels, class_names



def main():

    rootdir = "hw-4-images/training_set"
    images, labels, class_names = iterate_through_directories(rootdir)


    print("num images:", len(images), "num labels:", len(labels))

    print("classes", class_names)
    print("classes", len(class_names))

    preprocessed_images = [parse_image(img) for img in images]

    model = keras.Sequential()

    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(15, activation="softmax"))
    print("layers added to model")
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print(model)
    model.fit(preprocessed_images, labels, epochs=5)


if __name__ == "__main__":
    main()
