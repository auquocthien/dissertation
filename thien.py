# # Import TF and TF Hub libraries.
# import tensorflow as tf
# import tensorflow_hub as hub
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# # Load the input image.
# image_path = 'movenet\input_image.jpeg'
# image = tf.io.read_file(image_path)
# image = tf.compat.v1.image.decode_jpeg(image)
# image = tf.expand_dims(image, axis=0)
# # Resize and pad the image to keep the aspect ratio and fit the expected size.
# image = tf.cast(tf.image.resize_with_pad(image, 192, 192), dtype=tf.int32)

# # Download the model from TF Hub.
# model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
# movenet = model.signatures['serving_default']

# # Run model inference.
# outputs = movenet(image)
# # Output is a [1, 1, 17, 3] tensor.
# keypoints = outputs['output_0']

# print(keypoints)
import os

files = [file for file in os.listdir('./movenet/dataset')]

print(len(files))
