import matplotlib.pyplot as plt

def plot_logits(logits_r, tit):
    plt.subplot()
    plt.imshow(logits_r, origin = 'lower')

    plt.text(41, 70, r'$\leftarrow$ Decision Boundary', fontsize = 16)
    plt.text(48, 60, '"Zero-Crossing"', fontsize = 16)

    plt.colorbar()
    plt.contour(logits_r, 0, linewidths = 3, colors = 'red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(tit)
    plt.show()
