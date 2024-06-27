import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Parameters
p = 0.48  # probability of success

# Define the geometric distribution PMF
def geometric_pmf(k, p):
    return (1 - p)**(k - 1) * p

# Generate values
x = np.arange(1, 10)
y = geometric_pmf(x, p)

# Find the point where probability drops below 0.5%
threshold_point = None
for i, prob in enumerate(y):
    if prob < 0.005:
        threshold_point = i + 1
        break

# Calculate expectation and median
expectation = 1 / p
median = np.ceil(np.log(0.5) / np.log(1 - p))

# Plot the probability distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=x, y=y, color='blue', alpha=0.6)

# Mark the threshold point on the graph
plt.axvline(threshold_point-1, color='red', linestyle='--', label=f'P < 0.5% at k={threshold_point}')

# Mark the expectation and median on the graph
plt.axvline(expectation-1, color='green', linestyle='-', label=f'Expectation (mean) = {expectation:.2f}')
plt.axvline(median-1, color='orange', linestyle='-', label=f'Median = {median}')
plt.xlabel('Number of Trials (k)')
plt.ylabel('Probability')
plt.title('Geometric Distribution (p = 0.48)')
plt.ylim(0, 0.6)
plt.grid(True)
plt.legend()
plt.show()

print(f'Expectation (mean) of the geometric distribution: {expectation}')
print(f'Median of the geometric distribution: {median}')
print(f'Probability drops below 0.5% at k = {threshold_point}')