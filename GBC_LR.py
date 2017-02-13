import csv
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import normalize
from sklearn.preprocessing import OneHotEncoder
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
print(len(headers))
train = np.array(train)
train_X = train[:, :-1]
train_y = train[:, -1]

# Spit the data set into roughly 2 halfs
# This allows us to not overfit with stacking
train_x1 = train_X[:30000]
train_x2 = train_X[30000:]


train_y1 = train_y[:30000]
train_y2 = train_y[30000:]

# We are first going to use Graident Boosting to transform
# the data and then using One Hot Encoding.
# After this, we will then try and fit a Logist Regression
grd = GradientBoostingClassifier()
grd_enc = OneHotEncoder()
grd.fit(train_x1, train_y1)
grd_enc.fit(grd.apply(train_x1)[:,:, 0])

grd_lm = LogisticRegression(penalty = 'l2', C = .0115)


grd_lm.fit(grd_enc.transform(grd.apply(train_x2)[:,:, 0]), train_y2)



#Import test data
test_x = []
with open('test_2012.csv', 'r') as f:
	first_row = f.readline()
	headers = first_row.split(',')
	for row in f:
		ints = [int(elem) for elem in row.split(',')]
		test_x.append(ints)
f.close()
print(len(headers))
test_x = np.array(test_x)

# Apply the model to the test data

test_y = grd_lm.predict(grd_enc.transform(grd.apply(test_x)[:, :, 0]))

# Write the output of the prediction to a csv file
with open('GBC_LR_2012.csv', 'w', newline='') as f:
	output = csv.writer(f)
	output.writerow(['id', 'PES1'])
	for i in range(len(test_x)):
		output.writerow([round(test_x[i, 0]), round(test_y[i])])






