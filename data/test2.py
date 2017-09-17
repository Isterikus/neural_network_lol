from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
import numpy

batch_size = 50 # in each iteration, we consider 128 training examples at once
num_epochs = 20 # we iterate twenty times over the entire training set
hidden_size = 10 # there will be 512 neurons in both hidden layers

num_train = 60000 # there are 60000 training examples in MNIST
num_test = 10000 # there are 10000 test examples in MNIST

# wr00, wr01, wr02, wr03, wr04, wr10, wr11, wr12, wr13, wr14 <- input
# x: [[10 wrs], [10 wrs], ...]
wr1 = [0.43 + (i / 100) for i in range(10)]
wr2 = [0.53 - (i / 100) for i in range(10)]

X_train = numpy.array([wr1] * 1000 + [wr2] * 1000)
X_test = numpy.array([wr2] * 10 + [wr1] * 10)

# y: [[bool], [bool], ...]

Y_train = numpy.array([1] * 1000 + [0] * 1000)
Y_test = numpy.array([0] * 10 + [1] * 10)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

Y_train = Y_train.astype("float32")
Y_test = Y_test.astype("float32")
Y_train2 = np_utils.to_categorical(Y_train, num_classes = 2) # One-hot encode the labels
Y_test2 = np_utils.to_categorical(Y_test, num_classes = 2) # One-hot encode the labels

inp = Input(shape=(10,)) # Our input is a 1D vector of size 10
hidden_1 = Dense(hidden_size, activation='relu')(inp) # First hidden ReLU layer
hidden_2 = Dense(hidden_size, activation='relu')(hidden_1) # Second hidden ReLU layer
out = Dense(2, activation='sigmoid')(hidden_2) # Output softmax layer

model = Model(input=inp, output=out) # To define a model, just specify its input and output layers

model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy


print(X_train)
print('===') 
print(Y_train2)

model.fit(X_train, Y_train2, batch_size=batch_size, epochs=num_epochs)
loss_and_metrics = model.evaluate(X_test, Y_test2, batch_size=batch_size) # Evaluate the trained model on the test set!

print(loss_and_metrics)


