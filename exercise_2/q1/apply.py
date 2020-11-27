import numpy as np
from linclass import linclass
from leastSquares import leastSquares
from plot_ import plot_

train = {}
test = {}
## Load the data
train.update({'data': np.loadtxt('lc_train_data.dat')})
train.update({'label': np.loadtxt('lc_train_label.dat')})
test.update({'data': np.loadtxt('lc_test_data.dat')})
test.update({'label': np.loadtxt('lc_test_label.dat')})

## Q1 a) Train the classifier using the training dataset
weight, bias = leastSquares(train['data'], train['label'])

## Q1 b) Evaluate the classifier on the training dataset
train.update({'prediction': linclass(weight, bias, train['data'])})

# Print and show the performance of the classifier
train.update({'acc' : sum(train['prediction'] == train['label'])/len(train['label'])})
print('Accuracy on train set: {0}'.format(train['acc']))
plot_(train['data'], train['label'], weight, bias, 'Train Set')


# Test the classifier on the test dataset
test.update({'prediction': linclass(weight, bias, test['data'])})

# Print and show the performance of the classifier
test.update({'acc' : sum(test['prediction'] == test['label'])/len(test['label'])})
plot_(test['data'], test['label'], weight, bias, 'Test Set')
print('Accuracy on test set: \t {0}\n'.format(test['acc']))


## Bonus: Add outlier to training data, what happens?
print('Adding outliers...')

train['data'] = np.append(train['data'], [[1.5, -0.4],[1.45, -0.35]], axis = 0)
train['label'] = np.append(train['label'], [[-1],[-1]])

# Train the classifier using the training dataset
weight, bias = leastSquares(train['data'], train['label'])

# Evaluate the classifier on the training dataset
train['prediction'] = linclass(weight, bias, train['data'])

# Print and show the performance of the classifier
train['acc'] = sum(train['prediction'] == train['label'])/len(train['label'])
print('Accuracy on train set: {0}'.format(train['acc']))
plot_(train['data'], train['label'], weight, bias, 'Train Set')


# Test the classifier on the test dataset
test['prediction'] = linclass(weight, bias, test['data'])

# Print and show the performance of the classifier
test['acc'] = sum(test['prediction']==test['label'])/len(test['label'])
print('Accuracy on test set: \t {0}\n'.format(test['acc']))
plot_(test['data'], test['label'], weight, bias, 'Test Set')