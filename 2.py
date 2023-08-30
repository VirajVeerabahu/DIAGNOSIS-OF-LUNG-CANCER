# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 14:03:39 2022

@author: DLK
"""

import numpy as np
from keras.models import load_model
from keras.preprocessing import image

class cancer:
    def __init__(self,filename):
        self.filename =filename


    def predictiondogcat(self):
        # load model
        model = load_model('modelforcancer.h5')

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)

        if result[0][0] == 1:
            prediction = 'Malignant case'
            return [{ "image" : prediction}]
        else:
            prediction = 'Normal case'
            return [{"image": prediction}]





