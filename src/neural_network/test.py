from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from keras.models import model_from_json
import numpy

'''

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

'''
data_path = "../../data/lol/"

data = {}
c_champs_count = 10
# if (out == 1) ? (RED TEAM WIN) || (BLUE TEAM WIN)
with open(data_path + "winRates_commandWin.psv") as fin:
	strs = fin.read.split('\n')
	rng = int(len(strs) / 0.75)
	for i in range(strs):
		data[dt]['in'] = []
		data[dt]['out'] = []
		dt = (i < rng) ? 'train' : 'test'
		data[dt]['in'].append(strs.split(' ')[:-2])
		data[dt]['out'].append(strs.split(' ')[c_champs_count:])

# wr1 = [0.43 + (i / 100.0) for i in range(10)]
# wr2 = [0.53 - (i / 100) for i in range(10)]

# print(wr1)
# print(wr2)
	
# [[10 floats], [10 floats]]
X_train = numpy.array([float(i) for i in data['train']['in']] + [int(i) for i in data['train']['in']])
X_test = numpy.array([float(i) for i in data['test']['in']] + [int(i) for i in data['test']['in']])

# X_train = numpy.array(data['train'])

# y: [[bool], [bool], ...]

Y_train = numpy.array([float(i) for i in data['train']['out']] + [int(i) for i in data['train']['out']])
Y_test = numpy.array([float(i) for i in data['test']['out']] + [(i) for i in data['train']['out']])




X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

Y_train = Y_train.astype("float32")
Y_test = Y_test.astype("float32")
Y_train2 = np_utils.to_categorical(Y_train, num_classes = 2) # One-hot encode the labels
Y_test2 = np_utils.to_categorical(Y_test, num_classes = 2) # One-hot encode the labels

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
 
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X_test, Y_test2)
print(score)