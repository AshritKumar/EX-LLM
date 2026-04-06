import numpy as np

arr = [[-1, 7, 8],
       [3, 6, 0],
       [-6, 4, -2]]

npArr = np.array(arr)

min = np.min(arr)
max = np.max(arr)

print("whole min ", min)
print("whole max ", max)

###################################
print("\n\n")
axis0Min = np.min(arr, axis=0) # axis0 goes through columns (across rows)
axis0Max = np.max(arr, axis=0) # axis0 goes through columns (across rows)

print("axis 0 min ", axis0Min)
print("axis 0 max ", axis0Max)

##########################################

# will work with normal list of lists or np Array
print("\n\n")
axis1Min = np.min(npArr,axis=1) # axis1 goes through rows (across columns)
axis1Max = np.max(npArr, axis=1) # axis1 goes through rows (across columns)

print("axis 1 min ", axis1Min)
print("axis 1 max ", axis1Max)


############################################
###### ARG MIN/MAX ######################
###### Returns indices of min/max values along specified axis
print("\n\n############### ARG MIN AND MAX ####################\n\n")
argMin = np.argmin(arr) # returns the position of the element in the array, 6 in this case (-6 position is 6)
argMax = np.argmax(arr) # returns 2nd position

print("argmin ", argMin)
print("argmax ", argMax)

argMinAxis0 = np.argmin(arr, axis=0)
argMaxAxis0 = np.argmax(arr, axis=0)

print("\n\n")

print("argmin axis 0", argMinAxis0)
print("argmax axis 0", argMaxAxis0)

argMinAxis1 = np.argmin(arr, axis=1)
argMaxAxis1 = np.argmax(arr, axis=1)

print("\n\n")

print("argmin axis 1", argMinAxis1)
print("argmax axis 1", argMaxAxis1)