import numpy as np
import matplotlib.pyplot as plt
from parameters import parameters
from svmlin import svmlin
C, C2, norm = parameters()

# Q2 b) USPS training
print('Training SVM on USPS digit dataset')
one = {}
three = {}
eight = {}
# Load digit data
one.update({'train': np.loadtxt('digit_1_train.dat', delimiter=',')})
one.update({'test': np.loadtxt('digit_1_test.dat', delimiter=',')})
three.update({'train': np.loadtxt('digit_3_train.dat', delimiter=',')})
three.update({'test': np.loadtxt('digit_3_test.dat', delimiter=',')})
eight.update({'train': np.loadtxt('digit_8_train.dat', delimiter=',')})
eight.update({'test': np.loadtxt('digit_8_test.dat', delimiter=',')})

# Visualize data
img = three['train'][24, :]
img = np.reshape(img, (16, 16))
plt.subplot()
plt.imshow(img)
plt.show()


# Concatenate 1 and 3
X = np.concatenate((one['train'], three['train']))  # concatenate 1 and 3 train data. Shape: (N x F)
t_train1 = np.ones(one['train'].shape[0])  # digit 1 is positive class
t_train3 = np.ones(three['train'].shape[0])*(-1)  # digit 3 is negative class
t = np.concatenate((t_train1, t_train3))  # concatenate 1 and 3 labels data
print(X, t, C)
# If you want to retrain, use the following two lines _instead_ of the load
_, sv, w, b, result, _ = svmlin(X, t, C)
np.savez('alpha_1_vs_3.npz', w, b, result, sv)
npz = np.load('alpha_1_vs_3.npz')
w = npz['arr_0']
b = npz['arr_1']
result = npz['arr_2']
sv = npz['arr_3']

# Evaluate
train_result = result
train_result[train_result < 0] = -1
train_result[train_result >= 0] = 1
train_error = len(train_result[train_result != t])
print(' ')
print('*** 1-vs-3 ***')
print('Number of SV: {0}\n'.format(sum(sv)))
print('Width of margin: {0}\n'.format(2 / np.linalg.norm(w)))
print('Train Error 1_vs_3: ', train_error)

Xtest = np.concatenate((one['test'], three['test']))
ttest = np.concatenate((np.ones(one['test'].shape[0]), (-1) * np.ones(three['test'].shape[0])))

testresult = Xtest.dot(w) + b

testresult[testresult < 0] = -1
testresult[testresult >= 0] = 1
test_error = len(testresult[testresult != ttest]) / len(ttest)
print('Test Error 1_vs_3: ', str(test_error))
 
# Visualize the weight vector W
# positive values(in red) gives shape of 1
# negative values(in blue) gives shape of 3
w = np.reshape(w, (16, 16))
plt.subplot()
plt.imshow(w)
plt.show()

# Q2 c) Classify digits 3 and 8
# train error 0.2# on _full_ dataset
# test errors vary largely by choice of test set... thus, there is no
# "correct" number.

X = np.concatenate((three['train'], eight['train']))  # concatenate 3 and 8 train data
t_train3 = np.ones(three['train'].shape[0])  # digit 3 is positive class
t_train8 = np.ones(eight['train'].shape[0]) * (-1)  # digit 8 is negative class
t = np.concatenate((t_train3, t_train8))  # concatenate 3 and 8 label data

# If you want to retrain use the following two lines _instead_ of the load
_, sv, w, b, result, _ = svmlin(X, t, C)
np.savez('alpha_3_vs_8.npz', w, b, result, sv)
npz = np.load('alpha_3_vs_8.npz')
w = npz['arr_0']
b = npz['arr_1']
result = npz['arr_2']
sv = npz['arr_3']


# Evaluate
train_result = result
train_result[train_result < 0] = -1
train_result[train_result >= 0] = 1
train_error = len(train_result[train_result != t])
print(' ')
print('*** 3-vs-8 ***')
print('Number of SV: {0}\n'.format(sum(sv)))
print('Width of margin: {0}\n'.format(2/np.linalg.norm(w)))
print('Train Error 3-vs-8: ', train_error)

Xtest = np.concatenate((three['test'], eight['test']))
ttest = np.concatenate((np.ones(three['test'].shape[0]), (-1)*np.ones(eight['test'].shape[0])))

testresult = Xtest.dot(w) + b

testresult[testresult < 0] = -1
testresult[testresult >= 0] = 1
test_error = len(testresult[testresult != ttest]) / len(ttest)
print('Test Error 3-vs-8: ', test_error)

# Visualize the weight vector W
# positive values(in red) gives shape of 3
# negative values(in blue) gives shape of 8
plt.subplot()
w = np.reshape(w, (16, 16))
plt.subplot()
plt.imshow(w)
plt.show()
# 
# train error 12 on _full_ dataset
# test errors vary largely by choice of test set... thus, there is no
# "correct" number.
