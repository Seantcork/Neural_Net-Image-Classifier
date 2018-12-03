import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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

    # walk returns 3-part tuple in format (directory, [subdirs], [files])
    iterator = tf.gfile.Walk(directory)

    classes = next(iterator)[1]

    labels = []
    file_paths = []
    imgs = []

    for subdir_name in classes:
        dir_path = rootdir + '/' + subdir_name
        subdir = os.fsencode(dir_path)
        generator = tf.gfile.Walk(subdir)
        files = next(generator)[2]
        for file_name in files:
            file_path = dir_path + '/' + file_name
            img = tf.read_file(file_path)

            file_paths.append(file_path)
            imgs.append(img)
            labels.append(subdir_name)

    return classes, labels, file_paths, imgs




def main():

    rootdir = "training_set"

    classes, labels, file_paths, imgs = iterate_through_directories(rootdir)

    print("done creating images")
    print(imgs[0])
    decoded = tf.image.decode_jpeg(imgs[0])
    resized = tf.image.resize_images(decoded, [28, 28])
    greyscale = tf.image.rgb_to_grayscale(resized)


    # new_imgs = [parse_image(img) for img in imgs]
    # print("done parsing images")
    #
    # model = keras.Sequential([
    #     keras.layers.Flatten(input_shape=(28, 28)),
    #     keras.layers.Dense(1, activation=tf.nn.relu),
    #     keras.layers.Dense(15, activation=tf.nn.softmax)])
    # model.compile(optimizer=tf.train.AdamOptimizer(),
    #               loss='sparse_categorical_crossentropy',
    #               metrics=['accuracy'])
    # model.fit(new_imgs, labels, epochs=5, steps_per_epoch=5)


if __name__ == "__main__":
    main()
