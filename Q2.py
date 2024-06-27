import numpy as np
import scipy.integrate as integrate
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# Constants
xi5 = 0.77
xi6 = 3
xi7 = 0.22
xi8 = 6

# Define the PDF
def pdf(y):
    return 4.62 * y * np.exp(-3 * y**2) + 10.56 * y**7 * np.exp(-6 * y**8)

# Probability between 2 and 4 hours
prob_2_4, _ = integrate.quad(pdf, 2, 4)

# Mean (expected value)
mean, _ = integrate.quad(lambda y: y * pdf(y), 0, np.inf)

# Variance
var, _ = integrate.quad(lambda y: (y - mean)**2 * pdf(y), 0, np.inf)

# Calculate the CDF
def cdf(y):
    result, _ = integrate.quad(pdf, 0, y)
    return result

# Find quartiles by solving cdf(y) = p for p in [0.25, 0.5, 0.75]
quartiles = [root_scalar(lambda y: cdf(y) - p, bracket=[0, 10]).root for p in [0.25, 0.5, 0.75]]

# Assign quartile values
q1, median, q3 = quartiles

# Prepare data for PDF plot
y_values = np.linspace(0, 4, 1000)
pdf_values = pdf(y_values)

# Prepare histogram data
time_minutes = np.linspace(0, 240, 1000) / 60  # 0 to 240 minutes, converted to hours
hist_values = pdf(time_minutes)

# Data arrays for potential visualization
pdf_data = {
    'y_values': y_values,
    'pdf_values': pdf_values
}

hist_data = {
    'time_minutes': time_minutes * 60,  # converting back to minutes for histogram
    'hist_values': hist_values
}

# Summary statistics
summary_statistics = {
    'Probability (2 to 4 hours)': prob_2_4,
    'Mean': mean,
    'Variance': var,
    'Q1': q1,
    'Median': median,
    'Q3': q3
}

# Print summary statistics
print("Summary Statistics:")
for key, value in summary_statistics.items():
    print(f"{key}: {value}")

# Plotting PDF with mean, variance, and quartiles
plt.figure(figsize=(10, 6))
plt.plot(pdf_data['y_values'], pdf_data['pdf_values'], label='PDF')

# Plot mean, variance, and quartiles
plt.axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.2f}')
plt.axvline(q1, color='g', linestyle='--', label=f'Q1: {q1:.2f}')
plt.axvline(median, color='b', linestyle='--', label=f'Median: {median:.2f}')
plt.axvline(q3, color='m', linestyle='--', label=f'Q3: {q3:.2f}')

plt.xlabel('Time (hours)')
plt.ylabel('Probability Density')
plt.title('Probability Density Function of Waiting Time')
plt.legend()
plt.grid(True)
plt.show()

# Plotting Histogram
plt.figure(figsize=(10, 6))
plt.bar(hist_data['time_minutes'], hist_data['hist_values'], width=0.1, label='Histogram')
plt.xlabel('Time (minutes)')
plt.ylabel('Probability')
plt.title('Histogram of Waiting Time in Minutes')
plt.legend()
plt.grid(True)
plt.show()