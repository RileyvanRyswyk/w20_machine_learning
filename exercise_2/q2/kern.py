def kern(x1, x2, d):
    r = (x2.T.dot(x1) + 1) ** d
    return r
