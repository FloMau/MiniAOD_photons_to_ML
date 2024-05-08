import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import boost_histogram as bh
from tensorflow import keras

import myplotparams


path = '/home/tim/uni/master/master/cluster/'
### load the model
# modelname = 'cnn'
modelname = 'cnn_weighted2'
model = keras.models.load_model(path + modelname + '.keras')

model.compile(optimizer=model.optimizer,
              loss=model.loss,
              # metrics=model.metrics,
              weighted_metrics='accuracy'
              )
model.summary()

### load and prepare the data
trainingfiles = [path + 'training1.pkl', path + 'training1_rechits.npy']
from mynetworks import load_and_prepare_data
_, (x_test, y_test) = load_and_prepare_data(*trainingfiles, 0.1, use_preselection=False, use_log=False)



test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=0)
# print(test_loss)
print('weighted_accuracy:', test_acc)
