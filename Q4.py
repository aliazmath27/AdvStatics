import numpy as np
from scipy.stats import t

# Given data
xi11 = 981  # mean weight
xi12 = 21.3  # standard deviation
xi14 = np.array([1090, 1091, 948, 1125, 1110, 855, 945, 933, 1108, 1034])  # sample weights

# Sample statistics
n = len(xi14)
sample_mean = np.mean(xi14)
sample_std = np.std(xi14, ddof=1)

# Hypothesis testing
# Null Hypothesis H0: mu = 981
# Alternative Hypothesis H1: mu < 981

# Calculate t-statistic
t_stat = (sample_mean - xi11) / (sample_std / np.sqrt(n))

# Degrees of freedom
df = n - 1

# Critical value for one-tailed test at alpha = 0.05
alpha = 0.05
t_critical = t.ppf(alpha, df)

# P-value
p_value = t.cdf(t_stat, df)

# Decision rule
decision = "Reject H0" if t_stat < t_critical else "Fail to Reject H0"

# Print results
print(f'Sample Mean: {sample_mean:.2f}')
print(f'Sample Standard Deviation: {sample_std:.2f}')
print(f'T-statistic: {t_stat:.4f}')
print(f'Critical Value (t_critical) at alpha={alpha}: {t_critical:.4f}')
print(f'P-value: {p_value:.4f}')
print(f'Decision: {decision}')
