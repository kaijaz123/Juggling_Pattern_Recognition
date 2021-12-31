import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, LeakyReLU
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
from tensorflow.keras import metrics
import argparse
from data_generator import data_generator
import numpy as np

# add arg parser for command line interface
parser = argparse.ArgumentParser()
parser.add_argument('--loss', default = 'categorical_crossentropy', type=str, help='loss function to be used')
parser.add_argument('--lr', default = 1, type=float, help='learning rate')
parser.add_argument('--epochs', default = 20, type=int, help='epochs quantity')
parser.add_argument('--bs', default = 8, type=int, help='batch size')
parser.add_argument('--checkpoint', default = 'pattern_model.h5', type=str, help='path for model saving')
parser.add_argument('--train', default = True, type=bool, help='train model')
parser.add_argument('--test', default = True, type=bool, help='test model')

class Model:
    def __init__(self, loss, learning_rate, epochs, batch_size, model_filename):
        # initialization
        self.loss = loss
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_filename = model_filename

        # get data information
        data = data_generator(12600) # number of generate data
        data.generate_data()
        self.x_train = data.x_train
        self.x_test = data.x_test
        self.y_train = data.y_train
        self.y_test = data.y_test

        #build model
        self.build_model()

    def model_structure(self):
        input_shape = self.x_train.shape[1:]
        output_shape = self.y_train.shape[1]
        # build model
        model = Sequential(name = 'pattern_recognition_model')
        model.add(Dense(120, input_shape = input_shape))
        model.add(LeakyReLU())
        model.add(Dense(units=120, kernel_regularizer=regularizers.l2(0.0001)))
        model.add(LeakyReLU())
        model.add(Dense(units=60, kernel_regularizer=regularizers.l2(0.0001)))
        model.add(LeakyReLU())
        model.add(Dense(output_shape, activation = 'softmax'))

        return model

    def build_model(self):
        self.model = self.model_structure()
        print(self.model.summary())

    def train_model(self):
        # compile model
        opt = optimizers.Adam(learning_rate = self.learning_rate)
        checkpoints = ModelCheckpoint(self.model_filename, monitor = 'val_accuracy', verbose=1, save_best_only=True, period=1)
        self.model.compile(optimizer = opt, loss = self.loss, metrics = ['accuracy'])
        self.model.fit(x = self.x_train,
                       y = self.y_train,
                       batch_size = self.batch_size,
                       validation_data = (self.x_test,self.y_test),
                       epochs = self.epochs,
                       shuffle = True,
                       callbacks = [checkpoints])

    def model_loading(self):
        self.loaded_model = load_model(self.model_filename)

    def test_model(self):
        # load model
        self.model_loading()

        # evaluation on x_test
        metrics_score = self.loaded_model.evaluate(self.x_test,self.y_test)
        print("Metrics loss: {}".format(metrics_score[0]))
        print("Metrics accuracy: {}".format(metrics_score[1]))

        # prediction
        y_predict = self.loaded_model.predict(self.x_test)
        y_pred = np.argmax(y_predict, axis = -1)
        print(y_pred)
        print("y true:",self.y_test)

if __name__ == "__main__":
    args = parser.parse_args()
    # define model
    model = Model(loss = args.loss, learning_rate = args.lr, epochs = args.epochs, batch_size = args.bs, model_filename = args.checkpoint)
    if args.train:
        model.train_model()
    if args.test:
        model.test_model()
