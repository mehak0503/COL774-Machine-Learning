from keras.models import Sequential
from keras.layers import Dense
import numpy

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("train_x.csv", delimiter=",")
datasety = numpy.loadtxt("train_y.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,:]
datasety = datasety.reshape(-1,1)
Y = datasety[:,0]
# create model
model = Sequential()
model.add(Dense(250, input_dim=250, init='uniform', activation='sigmoid'))
model.add(Dense(1000, init='uniform', activation='sigmoid'))
model.add(Dense(1, init='uniform', activation='tanh'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10,  verbose=2)
# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)


