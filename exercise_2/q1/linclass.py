def linclass(weight, bias, data):
    # Linear Classifier
    #
    # INPUT:
    # weight      : weights                (dim x 1)
    # bias        : bias term              (scalar)
    # data        : Input to be classified (num_samples x dim)
    #
    # OUTPUT:
    # class_pred       : Predicted class (+-1) values  (num_samples x 1)

    # Perform linear classification i.e. class prediction
    Y = data @ weight + bias

    # classify around 0
    class_pred = []
    for result in Y:
        if result >= 0:
            class_pred.append(1)
        else:
            class_pred.append(-1)

    return class_pred


