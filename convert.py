import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("./models/tensorflow/nanodet_plus")
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
open("./models/tflite/nanodet_plus.tflite", "wb").write(tflite_model)