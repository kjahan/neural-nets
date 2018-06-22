import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

#Load iris dataset and properly encode Y (class)
def load_data(fn):
	# fix random seed for reproducibility
	seed = 10
	np.random.seed(seed)

	#load dataset
	dataframe = pd.read_csv(fn, header=None)
	df = dataframe.sample(frac=1)	#randomly shuffle dataframe
	dataset = df.values	#Get numpy array out of dataframe

	X = dataset[:,0:4].astype(float)
	Y = dataset[:,4]

	#encode class values as integers
	encoder = LabelEncoder()
	encoder.fit(Y)
	encoded_Y = encoder.transform(Y)
	#convert integers to dummy variables (i.e. one hot encoded)
	dummy_y = np_utils.to_categorical(encoded_Y)

	return X, dummy_y


#Define baseline model - Neural network topology: 4 inputs -> [10 hidden nodes] -> [10 hidden nodes] -> 3 outputs
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(10, input_dim=4, activation='relu'))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


def run(fn):
	X, dummy_y = load_data(fn)
	estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
	kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
	results = cross_val_score(estimator, X, dummy_y, cv=kfold)
	print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

    
if __name__ == "__main__":
	if len(sys.argv) < 2:
		print "Usage: python iris_nn_classifier.py dataset-filename"
		sys.exit(1)
	run(sys.argv[1])
