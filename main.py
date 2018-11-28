import tensorflow as tf

file = "training_set/animal/1966.17.13.a.jpg"
img = tf.image.decode_jpeg(file)
print(img)

