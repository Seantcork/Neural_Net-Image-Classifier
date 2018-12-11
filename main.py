import tensorflow as tf
from tensorflow import keras
import os
import numpy as np


from tensorflow.keras.preprocessing.image import ImageDataGenerator

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


# def iterate_through_directories(rootdir):
#     directory = os.fsencode(rootdir)

#     # walk returns 3-part tuple in format (directory, [subdirs], [files])
#     iterator = tf.gfile.Walk(directory)

#     classes = next(iterator)[1]

#     labels = []
#     file_paths = []
#     imgs = []

#     for subdir_name in classes:
#         dir_path = rootdir + '/' + subdir_name
#         subdir = os.fsencode(dir_path)
#         generator = tf.gfile.Walk(subdir)
#         files = next(generator)[2]
#         for file_name in files:
#             file_path = dir_path + '/' + file_name
#             if not file_name.endswith('.jpg'):
# 		continue
#             img = tf.read_file(file_path)
#             decoded = tf.image.decode_jpeg(img)
#             resized = tf.image.resize_images(decoded, [28, 28])
#             img = tf.image.rgb_to_grayscale(resized)
#             #print(tf.Session().run(tf.shape(img)))

#             file_paths.append(file_path)
#             imgs.append(img)
#             labels.append(subdir_name)

#     return classes, labels, file_paths, imgs



def iterate_through_directories2(rootdir):
	train_images = []
	for subdir, dirs, files in os.walk(rootdir):
		for file in files:
			img = tf.read_file(file)
			decoded = tf.image.decode_jpeg(img)
			resized = tf.image.resize_images(decoded, [28, 28])
			greyscale = tf.image.rgb_to_grayscale(resized)
			train_images.append(np.array(img), )

def main():
    #rootdir = "training_set"
    #classes, labels, file_paths, images = iterate_through_directories(rootdir)

    #img = images[0]
    #decoded = tf.image.decode_image(img)


    #decoded_imgs = images
    print("done decoding")


    # dataset = tf.data.Dataset.from_tensors((decoded_imgs, labels))
    # print(dataset.output_shapes)

    # #dataset = dataset.batch(32).repeat()

    # model = keras.Sequential([
    #     keras.layers.Flatten(input_shape=(1503, 28, 28)),
    #     keras.layers.Dense(1, activation=tf.nn.relu),
    #     keras.layers.Dense(15, activation=tf.nn.softmax)])
    img_input = keras.layers.Input(shape=(150, 150, 3))
    # First convolution extracts 16 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    x = keras.layers.Conv2D(16, 3, activation='relu')(img_input)
    x = keras.layers.MaxPooling2D(2)(x)

    # Second convolution extracts 32 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    x = keras.layers.Conv2D(32, 3, activation='relu')(x)
    x = keras.layers.MaxPooling2D(2)(x)

    # Third convolution extracts 64 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    x = keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = keras.layers.MaxPooling2D(2)(x)


    x = keras.layers.Flatten()(x)

    # Create a fully connected layer with ReLU activation and 512 hidden units
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dense(512, activation='relu')(x)


    output = keras.layers.Dense(14, activation='softmax')(x)

    model = keras.Model(img_input, output)
    # model = tf.keras.models.Sequential
    # tf.keras.layers.Flatten()
    # tf.keras.layers.Dense(512, activation='relu')(x)



    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    train_datagen = ImageDataGenerator(rescale=1./255)
    train_dir = "training_set"
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), class_mode='categorical', color_mode= 'rgb')
    test_datagen = ImageDataGenerator(rescale=1./255)
    validation_dir = "test_data"
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        class_mode='sparse', color_mode='rgb')
    print("dont with images")

    #model.fit(dataset, epochs=5, steps_per_epoch=5)



    history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,  # 2000 images = batch_size * steps
      epochs=20,validation_data=validation_generator,validation_steps=100)
    acc = history.history['acc']
    print(acc)

main()
