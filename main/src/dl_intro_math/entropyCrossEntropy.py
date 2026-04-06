import numpy as np
def entropyExample1():
    # we would have to consider both the probabilites of an event happening and not happening
    # always should sump up to one
    # say prpability of an event happening peh = 0.25
    peh = 0.25
    p = [peh, 1 - peh]
    H = 0
    for pi in p:
        H += -(pi * np.log(pi))
        # or
        # H -= pi * np.log(pi)  # equivalent calculation
    print("Entropy", H)

    # if we have just two events, then we can also write H explicitely
    # Also called as binary cross entropy
    H_explicit = -(peh * np.log(peh) + (1 - peh) * np.log(1 - peh))
    print("Explicit Entropy", H_explicit)


def exampleCrossEntropy():
    p = [1,0]
    q = [0.25, 0.75]
    H = 0

    for i in range(len(p)):
       H -= p[i] * np.log(q[i])
    
    H_explicit = -(p[0] * np.log(q[0]) + p[1] * np.log(q[1]))
    
    print("Cross Entropy", H)
    print("Explicit Cross Entropy", H_explicit)

    # in case p is a binary event like either/or, we can compute cross-entropy directly
    H_direct = -np.log(q[0])  # since p[0] = 1, p[1] = 0

    print("Direct Cross Entropy", H_direct)

if __name__ == "__main__":
    # entropyExample1()
    exampleCrossEntropy()