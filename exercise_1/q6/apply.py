import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from parameters import parameters
from getLogLikelihood import getLogLikelihood
from estGaussMixEM import estGaussMixEM
from EStep import EStep
from MStep import MStep
from regularize_cov import regularize_cov
from plotModes import plotModes
from skinDetection import skinDetection

epsilon, K, n_iter, skin_n_iter, skin_epsilon, skin_K, theta = parameters()


def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out


print('Question: Expectation Maximization Algorithm for GMMs')

# path
dirname = os.path.dirname(__file__)

# load datasets
data = [[], [], []]
data[0] = np.loadtxt(os.path.join(dirname, 'data1'))
data[1] = np.loadtxt(os.path.join(dirname, 'data2'))
data[2] = np.loadtxt(os.path.join(dirname, 'data3'))

# test getLogLikelihood
print('(a) testing getLogLikelihood function')
weights = [0.341398243018411, 0.367330235091507, 0.291271521890082]
means = [
    [3.006132088737974,  3.093100568285389],
    [0.196675859954268, -0.034521603109466],
    [-2.957520528756456,  2.991192198151507]
]
covariances = np.zeros((2, 2, 3))
covariances[:, :, 0] = [
    [0.949104844872119, -0.170637132238246],
    [-0.170637132238246,  2.011158266600814]
]
covariances[:, :, 1] = [
    [0.837094104536474, 0.044657749659523],
    [0.044657749659523, 1.327399518241827]
]
covariances[:, :, 2] = [
    [1.160661833073708, 0.058151801834449],
    [0.058151801834449, 0.927437098385088]
]

loglikelihoods = [-1.098653352229586e+03, -1.706951862352565e+03, -1.292882804841197e+03]
for idx in range(3):
    ll = getLogLikelihood(means, weights, covariances, data[idx])
    diff = loglikelihoods[idx] - ll
    print('LogLikelihood is {0}, should be {1}, difference: {2}\n'.format(ll, loglikelihoods[idx], diff))

# test EStep
print('\n')
print('(b) testing EStep function')
# load gamma values
testgamma = [[], [], []]
testgamma[0] = np.loadtxt(os.path.join(dirname, 'gamma1'))
testgamma[1] = np.loadtxt(os.path.join(dirname, 'gamma2'))
testgamma[2] = np.loadtxt(os.path.join(dirname, 'gamma3'))
for idx in range(3):
    _, gamma = EStep(means, covariances, weights, data[idx])
    absdiff = testgamma[idx] - gamma
    print('Sum of difference of gammas: {0}\n'.format(np.sum(absdiff)))

# test MStep
print('\n')
print('(c) testing MStep function')
# load gamma values
testparams = np.ndarray((3, 3), dtype=object)
# means
testparams[0, 0] = [
     [3.018041988488699,  3.101046000178649],
     [0.198328683921772, -0.019449541135746],
     [-2.964974332415026,  2.994362963328281]
]
testparams[0, 1] = [
     [3.987646604627858, -0.056285481712672],
     [0.064528352867431, -0.046345896337489],
     [-3.244342020825232,  0.164140465045744]
]
testparams[0, 2] = [
     [3.951117305917324, -0.913396187074355],
     [0.121144018117729, -0.040037587868608],
     [-3.054802211026562,  1.969195200268656]
]
# weights
testparams[1, 0] = [0.339408153353897, 0.370303288436004, 0.290288558210099]
testparams[1, 1] = [0.336051939551412, 0.432073585981995, 0.231874474466593]
testparams[1, 2] = [0.257806471569113, 0.379609598797200, 0.362583929633687]
# covariances
testparams[2, 0] = np.ndarray((2, 2, 3))
testparams[2, 0][:, :, 0] = [
     [0.928530520617187, -0.186093601749430],
     [-0.186093601749430,  2.005901936462142]
]
testparams[2, 0][:, :, 1] = [
    [0.838623744823879, 0.045317199218797],
    [0.045317199218797, 1.352200524531750]
]
testparams[2, 0][:, :, 2] = [
    [1.146594581079395, 0.064658231773354],
    [0.064658231773354, 0.925324018684456]
]
testparams[2, 1] = np.ndarray((2, 2, 3))
testparams[2, 1][:, :, 0] = [
     [0.333751473448182, -0.036902134347530],
     [-0.036902134347530,  0.249019229685320]
]
testparams[2, 1][:, :, 1] = [
    [2.790985903869931, 0.180319331359206],
    [0.180319331359206, 0.208102949332177]
]
testparams[2, 1][:, :, 2] = [
    [0.211697922392049, 0.052177894905363],
    [0.052177894905363, 0.221516522642614]
]
testparams[2,2] = np.ndarray((2, 2, 3))
testparams[2,2][:, :, 0] = [
     [0.258550175253901, -0.018706579394884],
     [-0.018706579394884,  0.102719055240694]
]
testparams[2,2][:, :, 1] = [
     [0.467180426168570, -0.153028946058116],
     [-0.153028946058116,  0.657684560660198]
]
testparams[2,2][:, :, 2] = [
    [0.559751011345552, 0.363911891484002],
    [0.363911891484002, 0.442160603656823]
]
for idx in range(3):
    weights, means, covariances, _ = MStep(testgamma[idx], data[idx])
    absmeandiff = abs(means - testparams[0, idx])
    absweightdiff = abs(weights - testparams[1, idx])
    abscovdiff = abs(covariances - testparams[2, idx])

    print('Sum of difference of means:       {0}\n'.format(np.sum(absmeandiff)))
    print('Sum of difference of weights:     {0}\n'.format(np.sum(absweightdiff)))
    print('Sum of difference of covariances: {0}\n'.format(np.sum(abscovdiff)))

# test regularization
print('\n')
print('(c) testing regularization of covariances')
regularized_cov = np.ndarray((2, 2, 3))
regularized_cov[:, :, 0] = [
     [0.938530520617187, -0.186093601749430],
     [-0.186093601749430,  2.015901936462142]
]
regularized_cov[:, :, 1] = [
    [0.848623744823879, 0.045317199218797],
    [0.045317199218797, 1.362200524531750]
]
regularized_cov[:, :, 2] = [
    [1.156594581079395, 0.064658231773354],
    [0.064658231773354, 0.935324018684456]
]
for idx in range(3):
    covariance = regularize_cov(testparams[2, 0][:, :, idx], 0.01)
    absdiff = abs(covariance - regularized_cov[:, :, idx])
    print('Sum of difference of covariances: {0}\n'.format(np.sum(absdiff)))


# compute GMM on all 3 datasets
print('\n')
print('(f) evaluating EM for GMM on all datasets')
'''for idx in range(3):
    print('evaluating on dataset {0}\n'.format(idx+1))

    # compute GMM
    weights, means, covariances = estGaussMixEM(data[idx], K, n_iter, epsilon)

    # plot result
    plt.subplot()
    plotModes(np.transpose(means), covariances, data[idx])
    plt.title('Data {0}'.format(idx+1))
    plt.show()'''

''''
# uncomment following lines to generate the result
# for different number of modes k plot the log likelihood for data3
num = 14
logLikelihood = np.zeros(num)
for k in range(num):
    # compute GMM
    weights, means, covariances = estGaussMixEM(data[2], k+1, n_iter, epsilon)
    logLikelihood[k] = getLogLikelihood(means, weights, covariances, data[2])

# plot result
plt.subplot()
plt.plot(range(num),logLikelihood)
plt.title('Loglikelihood for different number of k on Data 3')
plt.show()'''

# skin detection
print('\n')
print('(g) performing skin detection with GMMs')
sdata = np.loadtxt(os.path.join(dirname, 'skin.dat'))
ndata = np.loadtxt(os.path.join(dirname, 'non-skin.dat'))

img = im2double(misc.imread(os.path.join(dirname, 'faces.png')))

skin = skinDetection(ndata, sdata, skin_K, skin_n_iter, skin_epsilon, theta, img)
plt.imshow(skin)
plt.show()
misc.imsave(os.path.join(dirname, 'skin_detection.png', skin))
