import numpy as np
import matplotlib.pyplot as plt

def manualSoftMaxExample():
    z = [1,2,3]
    numarator = np.exp(z)
    denominator = np.sum(numarator)
    softmax = numarator / denominator
    print("Softmax probabilities:", softmax)
    print("Sum of probabilities:", np.sum(softmax))


def softMaxOverMultipleRandomValues():
    z = np.random.randint(-5, high=15, size=25)
    print(z)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator / denominator
    print("Softmax probabilities:", softmax)
    print("Sum of probabilities:", np.sum(softmax))

    # Plot the softmax values vs values
    plt.plot(z,softmax,'ko', alpha=0.7)
    plt.xlabel('Original number (z)')
    plt.ylabel('Softmaxified $\sigma$')
    # plt.yscale('log')
    plt.title('$\sum\sigma$ = %g' %np.sum(softmax))
    plt.show()

if __name__ == "__main__":
    # manualSoftMaxExample()
    softMaxOverMultipleRandomValues()