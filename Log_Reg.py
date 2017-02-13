import csv
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt 
from sklearn.preprocessing import normalize

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
train_X = train[:40000, :-1]
norm_X = normalize(train_X, axis = 0, norm = 'max')
train_y = train[:40000, -1]
valid_X = train[40000:, :-1]
norm_vX = normalize(valid_X, axis = 0, norm = 'max')
valid_y = train[40000:, -1]

# Perform logistic regression for feature selection
logreg_feat_sel = LogisticRegression(penalty = 'l1', C = 0.01)
logreg_feat_sel.fit(norm_X, train_y)
model = SelectFromModel(logreg_feat_sel, prefit = True)
X_new = model.transform(norm_X)
X_inds = model.get_support(indices = True)
vX_new = model.transform(norm_vX)

print(X_new.shape)
print(X_inds)
rel_cols = [headers[i] for i in X_inds]
print(rel_cols)

v_score = [0.] * 5
v_score = np.array(v_score)
# Perform logistic regression on the restricted list of features
# Modify penalty term and plot the cross-validation errors of each
for i in range(5):
	restr_logreg = LogisticRegression(penalty = 'l2', C = np.power(10., -i))
	restr_logreg.fit(X_new, train_y)
	logreg_pred_y = restr_logreg.predict(X_new)
	v_y = restr_logreg.predict(vX_new)
	v_score[i] = accuracy_score(valid_y, v_y)

plot_x = list(range(0, -5, -1))
plt.plot(plot_x, v_score)
plt.xlabel('log(C)')
plt.ylabel('Accuracy')
plt.show()



# Import test data
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
norm_test_x = normalize(test_x, axis = 0, norm = 'max')

# Apply the logistic  model to the test data
restr_test_x = model.transform(norm_test_x)
lr_test_y = restr_logreg.predict(restr_test_x)

# Write the output of the prediction to a csv file
with open('lr_test_2012.csv', 'w') as f:
	output = csv.writer(f)
	output.writerow(['id', 'PES1'])
	for i in range(len(test_x)):
		output.writerow([round(test_x[i, 0]), round(lr_test_y[i])])







