import matplotlib.pyplot as plt

def plot_(data, labels, weight, bias, window_title):
    #PLOT_ Summary of this function goes here
    #   Detailed explanation goes here

    # Define the range
    xmax = 1.5 #max(data(:, 1))
    xmin = -0.5 #min(data(:, 1))
    ymax = 1.5 #max(data(:, 2))
    ymin = -0.5 #min(data(:, 2))

    # Plot the data points and the decision line
    plt.subplot()
    plt.axis([xmin, xmax, ymin, ymax])
    plt.title(window_title)
    plt.plot(data[labels==1][:,0], data[labels==1][:,1], c = 'b', marker = 'x', linestyle='none', markersize=5)
    plt.plot(data[labels==-1][:,0], data[labels==-1][:,1], c = 'r', marker = 'o', linestyle='none', markersize=5, fillstyle='none')
    plt.plot([xmin, xmax], [-(weight[0]*xmin+bias)/weight[1], -(weight[0]*xmax+bias)/weight[1]], c = 'k')
    plt.show()
