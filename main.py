import tensorflow as tf



def parse_image(img, label):
    decoded = tf.image.decode_jpeg(img)
    resized = tf.image.resize_images(decoded, [28, 28])
    greyscale = tf.image.rgb_to_grayscale(resized)

    with tf.Session() as sess:
        # img1 = sess.run(resized)
        # print(img1.shape)
        # print(img1)

        greyscale = sess.run(greyscale)
        return greyscale, label



def main():
    # rootdir = "../test_set"
    file = "1967.14.11.a.jpg"
    img = tf.read_file(file)
    label = "Test"
    greyscale, label = parse_image(img, label)
    print(greyscale)

if __name__ == "__main__":
    main()
