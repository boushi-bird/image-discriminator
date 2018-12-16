from keras_app import KerasApp, preprocess_input, decode_predictions, target_size
import keras.preprocessing.image as PilImage
import numpy

class Discriminator:

  def __init__(self):
    self.model = KerasApp(weights='imagenet')

  def predict(self, file):
    image = PilImage.load_img(file, target_size=target_size)
    x = PilImage.img_to_array(image)
    x = numpy.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = self.model.predict(x)
    results = decode_predictions(preds, top=3)[0]
    return list(map(lambda result:{'name': result[1], 'ratio': numpy.float(result[2]) }, results))
