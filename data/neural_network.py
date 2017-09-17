from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from keras.models import model_from_json
import numpy

batch_size = 50 # in each iteration, we consider 128 training examples at once
num_epochs = 20 # we iterate twenty times over the entire training set
hidden_size = 20 # there will be 512 neurons in both hidden layers

num_train = 60000 # there are 60000 training examples in MNIST
num_test = 10000 # there are 10000 test examples in MNIST

id_to_wr = dict()
wr_to_id = dict()

with open('champId_winRate.psv') as f:
	for line in f:
		str_list = line.split(' ', 2)
		champId = int(str_list[0])
		winRate = float(str_list[1].replace('\n', ''))
		id_to_wr[champId] = winRate
		wr_to_id[winRate] = champId

with open("Big_Data.psv") as fin:
    with open("winRates_commandWin.psv", "w") as fout:
        for line in fin:
        	str_list = line.split(" ")[ :-1] # without \n
        	champIds = [int(s) for s in str_list[ :-1]] # without commandWin
        	commandWin = int(str_list[-1])

        	flag = True
        	for champId in champIds:
        		if id_to_wr.get(champId, None) is None:
        			flag = False
        			break
        	if flag == False:
        		continue

        	fout.write(" ".join(str(id_to_wr[i]) for i in champIds) + " " + str(commandWin) + '\n')

X_train = []
X_test = []

Y_train = []
Y_test = []

with open("winRates_commandWin.psv") as f:
	i = 0
	for line in f:
		str_list = line.split(" ")
		winRates = [float(s) for s in str_list[ :-1]] # without commandWin
		commandWin = [int(str_list[-1])]
		if ((i % 10) < 7):
			X_train.append(winRates)
			Y_train.append(commandWin)
		else:
			X_test.append(winRates)
			Y_test.append(commandWin)
		i += 1
	
# [[10 floats], [10 floats], ...]
X_train = numpy.array(X_train)
X_test = numpy.array(X_test)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

# y: [[index], [index], ...]

Y_train = numpy.array(Y_train)
Y_test = numpy.array(Y_test)

Y_train = Y_train.astype("float32")
Y_test = Y_test.astype("float32")

Y_train = np_utils.to_categorical(Y_train, num_classes = 2) # One-hot encode the labels
Y_test = np_utils.to_categorical(Y_test, num_classes = 2) # One-hot encode the labels



inp = Input(shape=(10,)) # Our input is a 1D vector of size 10
hidden_1 = Dense(hidden_size, activation='relu')(inp) # First hidden ReLU layer
hidden_2 = Dense(hidden_size, activation='relu')(hidden_1) # Second hidden ReLU layer
out = Dense(2, activation='sigmoid')(hidden_2) # Output softmax layer

model = Model(input=inp, output=out) # To define a model, just specify its input and output layers

model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy

model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs)



loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=batch_size) # Evaluate the trained model on the test set!

print "\nresult", loss_and_metrics[1]

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")


