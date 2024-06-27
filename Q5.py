import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Given data points
xi = np.array([-6, -3, -15, 11, 3, 19, -13, -8, -17, 6, 15, -5, 16, 18, 20, 1, 0, -12, -16, 7, 9, 13, -9, -18, -7, 14])
yi = np.array([111403322.94, 462282.09, -70724757197684.8, -5380035796270.05, -2044213.49, -3145974687829901, -11084186807773.54,
-13243758654.8, -358386303443529.75, -5082280322.49, -190295422421585.03, 54229059.24, -425014427378019.8, -1685432303696807, 
-5296482159859632, -3.11, 7.06, -4056762770084.19, -156908938442706.06, -30233030134.08, -508456232609.9, 
-34986514641390.84, -80538374028.83, -749346019952694.5, -1168081252.75, -87224755342309.27])

# Construct the design matrix
def design_matrix(x, degree):
    return np.vander(x, degree+1, increasing=True)

def cross_validate_ridge(X, y, lambdas, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    avg_mse = []

    for lambda_reg in lambdas:
        mse_list = []
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            XT = X_train.T
            I = np.eye(X_train.shape[1])
            alpha_ridge = np.linalg.inv(XT @ X_train + lambda_reg * I) @ XT @ y_train
            
            y_pred = X_val @ alpha_ridge
            mse = mean_squared_error(y_val, y_pred)
            mse_list.append(mse)
        
        avg_mse.append(np.mean(mse_list))
    
    return lambdas[np.argmin(avg_mse)], avg_mse

X = design_matrix(xi, 12)
XT = X.T

lambdas = np.logspace(-4, 4, 50)  # Range of lambda values to test

# Perform cross-validation
best_lambda, mse_values = cross_validate_ridge(X, yi, lambdas)

# OLS estimate
XT_X_inv = np.linalg.inv(XT @ X)
alpha_ols = XT_X_inv @ XT @ yi

# Ridge-regularized estimate
lambda_reg = .01  # Regularization parameter
I = np.eye(X.shape[1])
alpha_best_ridge = np.linalg.inv(XT @ X + best_lambda * I) @ XT @ yi

print(f'OLS estimates: {alpha_ols}')
print("Best Lambda:", best_lambda)
print("Ridge Coefficients with Best Lambda:", alpha_best_ridge)