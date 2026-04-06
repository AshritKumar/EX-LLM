import numpy as np

def manualMatrixMultiplication():
    # create two activation vectors
    # creates a vector of [0, 1, 2, 3, 4]
    acts_1 = np.arange(0,5)
    # creates a vector of [2, 3, 4, 5, 6]
    acts_2 = np.arange(2,7)

    # and two weights vectors
    weights_1 = np.linspace(-1,.4,len(acts_1))
    weights_2 = np.linspace(-.4,2,len(acts_2))**4

    # their dot products
    dp_1 = np.sum(acts_1*weights_1)
    dp_2 = np.sum(acts_2*weights_2)
    print(dp_1,dp_2)

    # and the cross-terms!
    dp_3 = np.sum(acts_1*weights_2)
    dp_4 = np.sum(acts_2*weights_1)

    print(dp_3,dp_4)

def numPyMatrixMultiplication():
    acts_1 = np.arange(0,5)
    acts_2 = np.arange(2,7)
    weights_1 = np.linspace(-1,.4,len(acts_1))
    weights_2 = np.linspace(-.4,2,len(acts_2))
    
    #  [[0 1 2 3 4]
    #  [2 3 4 5 6]]
    actsMatrix = np.array([acts_1, acts_2])
    print("actsMatrix ", actsMatrix)
    print("actsMatrix Shape:", actsMatrix.shape)

    weightsMatrix = np.array([weights_1, weights_2])
    print("weightsMatrix", weightsMatrix)
    print("weightsMatrix Shape:", weightsMatrix.shape)

    # We cannot directly multipley actsMatrix and weightsMatrx as they are of (2,5) and (2,5) shape. We need to do a transpose of either acts or weigths matrics
    result = actsMatrix @ weightsMatrix.T
    print("Result of actsMatrix @ weightsMatrix.T:\n", result)
    
    # We can also use np.dot
    npResult = np.dot(actsMatrix, weightsMatrix.T)
    print("Result of np.dot(actsMatrix, weightsMatrix.T):\n", npResult)

    

if __name__ == "__main__":
    # manualMatrixMultiplication()
    numPyMatrixMultiplication()