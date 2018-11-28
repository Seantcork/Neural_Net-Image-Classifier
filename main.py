import tensorflow as tf
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
    for path, dirs, files in os.walk(directory):

        base_name = os.path.basename(path)

        for file in files:
            img = tf.read_file(file)
            images.append(img)
            labels.append(base_name)

    return images, labels


def main():

    rootdir = "hw-4-images/training_set"
    images, labels = iterate_through_directories(rootdir)


    greyscale = parse_image(images[2])

    original_len = len(images)
    images = [parse_image(img) for img in images]

    print("original len", original_len)
    print("new len", len(images))




    # img = tf.read_file(file)
    # label = "Test"
    # greyscale, label = parse_image(img, label)
    # print(greyscale)

if __name__ == "__main__":
    main()
