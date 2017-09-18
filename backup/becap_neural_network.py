from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from keras.models import model_from_json
import numpy

batch_size = 50 # in each iteration, we consider 128 training examples at once
num_epochs = 20 # we iterate twenty times over the entire training set
hidden_size = 10 # there will be 512 neurons in both hidden layers

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
            ids = [int(s) for s in line.split(" ")[ :11]]
            if 0 in ids:
            	continue
            for i in ids[ :-1]:
                fout.write(str(id_to_wr[i]) + " ")
            teamWin = ids[-1]
            fout.write(str('0' if teamWin == 0 else '1 0') + '\n')


with open("winRates_commandWin.psv") as fin:
	fstrs = [s for s in fin.read().split('\n') if not s == '']
	rng = len(fstrs)
	r1 = int(rng / 0.60)
	r2 = rng - r1
	train_data = [fstrs[i] for i in range(rng) if i < r1]
	test_data = [fstrs[i] for i in range(rng) if not (i < r1)]

	with open("train_data.psv", "w") as fout:
		for s in train_data:
			fout.write(s + '\n')
	print("writed %d train_data.psv" % len(train_data))

	print(train_data)
	print("============================================")
	with open("test_data.psv", "w") as fout:
		for s in test_data:
			fout.write(s + '\n')
	print("writed %d test_data.psv")
print(test_data)

X_train = numpy.array([[float(s2) for s2 in s.split(' ')[ :-2]] for s in train_data])
X_test = numpy.array([[float(s2) for s2 in s.split(' ')[ :-2]] for s in test_data])

print(train_data[0])

Y_train = numpy.array([(0 if (int(i) for i in s.split(' ')[-2]) == 0 else 1) for s in train_data])
Y_test = numpy.array([(0 if (int(i) for i in s.split(' ')[-2]) == 0 else 1) for s in test_data])

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

model.fit(X_train, Y_train2, batch_size=batch_size, epochs=num_epochs)
#loss_and_metrics = model.evaluate(X_test, Y_test2, batch_size=batch_size) # Evaluate the trained model on the test set!

#print(loss_and_metrics)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")


