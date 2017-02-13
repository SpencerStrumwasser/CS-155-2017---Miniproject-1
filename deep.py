import numpy as np 
import csv
import tensorflow as tf
tf.python.control_flow_ops = tf
from sklearn.preprocessing import normalize
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout

train = []
headers = []
# Import training data
with open('train_2008.csv', 'r') as f:
    first_row = f.readline()
    headers = first_row.split(',')
    for row in f:
        ints = [int(elem) for elem in row.split(',')]
        train.append(ints)
f.close()
train = np.array(train)
train_X = train[:, :-1]
train_y = train[:, -1]
train_y = train_y - 1
train_X = normalize(train_X, norm='max', axis=0)
# one-hot encoding labels 
train_y = keras.utils.np_utils.to_categorical(train_y, 2)

model = Sequential()

model.add(Dense(120, input_shape=(382,)))
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(35))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('relu'))

model.add(Dense(2))
model.add(Activation('softmax'))

## Printing a summary of the layers and weights in your model
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['accuracy'])

fit = model.fit(train_X, train_y, batch_size=64, nb_epoch=20, verbose=1)

score = model.evaluate(train_X, train_y, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

with open('test_2008.csv', 'r') as dest_f:
    next(dest_f)
    data_iter = csv.reader(dest_f, delimiter = ',', quotechar='"')
    data = [data for data in data_iter]
test_data = np.asarray(data).astype(int)
X_test = normalize(test_data, norm='max', axis=0)

y_test = model.predict(X_test)
real_y_test = np.zeros(16000)
for i in range(16000):
    if int(round(y_test[i][0])) == 1:
        real_y_test[i] = 1
    else:
        real_y_test[i] = 2

with open('output.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id','PES1'])
    for i in range(16000):
        writer.writerow([i, int(round(real_y_test[i]))])


