import matplotlib.pyplot as plt

# ploy y = x^2
n = 10
t = -5
y = []
x = []

dy=[]

for i in range(n):
    x.append(t)
    y.append(t**2)

    dy.append(2*t)
    t+=1

print(x)
print(dy)

# plot

# Plot with proper formatting
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', label='y = x²', linewidth=2)
plt.plot(x, dy, 'r--', label='dy/dx = 2x', linewidth=2)

# Set axis labels
plt.xlabel('x')
plt.ylabel('y')

# Set axis limits for better view
# Option 1: Automatic with padding
# plt.xlim(min(x) - 1, max(x) + 1)
# plt.ylim(0, max(y) * 1.05)  # 5% padding above max

# Option 2: Custom ranges (uncomment to use)
# plt.xlim(0, 25)        # Focus on first 25 x values
# plt.ylim(0, 400)       # Focus on lower y values

# # Option 3: Different ranges for better comparison
# plt.xlim(-2, 32)       # More x padding
# plt.ylim(-20, 900)     # Show full range with padding

# Add grid for better readability
plt.grid(True, alpha=0.3)

# Add legend to distinguish the two lines
plt.legend()

# Add title
plt.title('Polynomial y = x² and its Derivative')

# Show the plot
plt.show()




