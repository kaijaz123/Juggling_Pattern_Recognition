import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, LeakyReLU
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
from tensorflow.keras import metrics
import argparse
import numpy as np

# add arg parser for command line interface
parser = argparse.ArgumentParser()
parser.add_argument('--loss', default = 'categorical_crossentropy', type=str, help='loss function to be used')
parser.add_argument('--lr', default = 1, type=float, help='learning rate')
parser.add_argument('--epochs', default = 20, type=int, help='epochs quantity')
parser.add_argument('--bs', default = 16, type=int, help='batch size')
parser.add_argument('--checkpoint', default = 'pattern_model.h5', type=str, help='path for model saving')
parser.add_argument('--train', default = True, type=bool, help='train model')
parser.add_argument('--test', default = True, type=bool, help='test model')
parser.add_argument('--x_data', default = 'data/x.npy', type=str, help='x data file name')
parser.add_argument('--y_data', default = 'data/y.npy', type=str, help='y data file name')

class Model:
    def __init__(self, loss, learning_rate, epochs, batch_size, model_filename, x_file, y_file):
        # initialization
        self.loss = loss
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_filename = model_filename

        # get data information
        self.data_loader(x_file, y_file)

        #build model
        self.build_model()

    def data_loader(self, x_file, y_file):
        x = np.load(x_file)
        y = np.load(y_file)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x,y, test_size = 0.2, random_state = 10)

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
    model = Model(loss = args.loss, learning_rate = args.lr, epochs = args.epochs, batch_size = args.bs, model_filename = args.checkpoint,
                  x_file = args.x_data, y_file = args.y_data)
    if args.train:
        model.train_model()
    if args.test:
        model.test_model()
