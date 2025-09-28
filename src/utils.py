import matplotlib.pyplot as plt

def stats(X, freq, xLabel=f'Runs', yLabel='Loss', title=f'Loss History'):
    plt.figure(figsize=(4, 4))
    plt.plot(X, freq, color='blue')
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.tight_layout()
    plt.show()