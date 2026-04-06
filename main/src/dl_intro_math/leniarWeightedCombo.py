import numpy as np
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline

matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

# Linear weighted combination example

def linear_weighted_combination_ex1(acts, weights):
    """
        Simple linear combination example.
    """

    linear_combination = np.dot(acts, weights)
    # we can also use sum
    linear_combination_from_sum = sum(acts*weights)
    print(f"Linear combination (dot product): {linear_combination}")
    print(f"Linear combination (sum): {linear_combination_from_sum}")
    return linear_combination

def linear_weighted_combination_ex12(acts, weights):
    """
    Example with repeating this with multiple random weights and random activations
    """
    print(f"avg of weights = {weights.mean():.3f}")
    print(f"avg of activations = {acts.mean():.3f}")

    num_samples = 10000

    # initialize (10k rows 2 cols)
    lin_combos = np.zeros((num_samples, 2))
    for sampI in range(num_samples):
        # itr1: fixed weigthts and random acts
        randActs = np.random.randn(len(weights))
        lin_combos[sampI,0] = np.sum(randActs * weights)

        #itr2: fixed activations and random weights
        randWeights = np.random.randn(len(weights))
        lin_combos[sampI,1] = np.sum(acts * randWeights)
    
    print(f"Mean of first column (fixed weights, random activations): {lin_combos[:,0].mean():.3f}")
    print(f"Mean of second column (fixed activations, random weights): {lin_combos[:,1].mean():.3f}")
    
    # plot histogram
    _, axs = plt.subplots(1, 3, figsize=(15, 3))
    axs[0].hist(lin_combos[:,0], bins=50, alpha=0.7)
    axs[0].hist(lin_combos[:,0],bins=50,color=[.9,.9,.9],edgecolor='gray')
    axs[0].axvline(lin_combos[:,0].mean(),color='r',linestyle='--',linewidth=3,label=f'Mean = {lin_combos[:,0].mean():.3f}')
    axs[0].set(xlabel='Linear weighted combination result',ylabel='Count',title='Fixed weights, random activations')
    axs[0].legend()

    axs[1].hist(lin_combos[:,1],bins=50,color=[.9,.9,.9],edgecolor='gray')
    axs[1].axvline(lin_combos[:,1].mean(),color='r',linestyle='--',linewidth=3,label=f'Mean = {lin_combos[:,1].mean():.3f}')
    axs[1].set(xlabel='Linear weighted combination result',ylabel='Count',title='Fixed activations, random weights')
    axs[1].legend()

    # plot a random normal distribution
    random_samples = np.random.randn(10000)
    axs[2].hist(random_samples, bins=50, color=[.9,.9,.9], edgecolor='gray')
    axs[2].axvline(random_samples.mean(), color='r', linestyle='--', linewidth=3, label=f'Mean = {random_samples.mean():.3f}')
    axs[2].set(xlabel='Random normal samples', ylabel='Count', title='Random normal distribution')
    axs[2].legend()

    plt.tight_layout()
    plt.show()

def linear_weighted_combination_with_bias(acts, weights, bias):
    """Compute linear combination with bias term."""
    num_samples = 10000
    lin_combos = np.zeros((num_samples,2))
    for sampI in range(num_samples):
        # Random activation vector
        randomActs = np.random.randn(len(weights))
        # This is not a correct way to add bias, this way will not shift the distribution
        lin_combos[sampI,0] = np.sum(randomActs * (weights + bias))
        
        # The correct way is to add bias after computing the weighted sum
        lin_combos[sampI,1] = np.sum(randomActs * weights) + bias
        
    # Plot now
    _, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    axs[0].hist(lin_combos[:,0], bins=50, color=[.9,.9,.9], edgecolor='gray')
    axs[0].axvline(lin_combos[:,0].mean(), color='r', linestyle='--', linewidth=3, label=f'Mean = {lin_combos[:,0].mean():.3f}')
    axs[0].set(xlabel='Linear weighted combination result', ylabel='Count', title='Incorrect bias addition')
    axs[0].legend()
    
    axs[1].hist(lin_combos[:,1], bins=50, color=[.9,.9,.9], edgecolor='gray')
    axs[1].axvline(lin_combos[:,1].mean(), color='r', linestyle='--', linewidth=3, label=f'Mean = {lin_combos[:,1].mean():.3f}')
    axs[1].set(xlabel='Linear weighted combination result', ylabel='Count', title='Correct bias addition')
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # random activations from previous layer
    acts = np.array([ 1,2,-5,3,0 ])
    # weights to current neuron
    weights = np.array([ .1,0,.3,-.1,.9 ])

    print("\n\nRunning linear_weighted_combination_ex1 with fixed values:")
    result = linear_weighted_combination_ex1(acts, weights)
    print(f"Returned value: {result}")

    # Example 2
    print("\n\nExample 2: Multiple random combinations")
    linear_weighted_combination_ex12(acts, weights)

    # Example 3- Introducing a bias (offset)
    print("\n\nExample 3: Linear combination with bias")
    linear_weighted_combination_with_bias(acts, weights, bias=3.0)
