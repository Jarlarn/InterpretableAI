import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ============================================================
# Task 1: Read and visualize the time series data
# ============================================================

training_data = np.loadtxt("A1_training.txt")
validation_data = np.loadtxt("A1_validation.txt")
test_data = np.loadtxt("A1_test.txt")

# Plot all three time series
plt.figure(figsize=(14, 5))
plt.subplot(1, 3, 1)
plt.plot(training_data)
plt.title("Training Data")
plt.xlabel("Time step")
plt.ylabel("Value")

plt.subplot(1, 3, 2)
plt.plot(validation_data, color="green")
plt.title("Validation Data")
plt.xlabel("Time step")
plt.ylabel("Value")

plt.subplot(1, 3, 3)
plt.plot(test_data, color="red")
plt.title("Test Data")
plt.xlabel("Time step")
plt.ylabel("Value")

plt.tight_layout()
plt.savefig("task1_time_series.png")
plt.show()

# ============================================================
# Task 2: Fit ARMA model and evaluate on test data
# ============================================================


def arma_model(phi, theta, mu, data):
    """
    General ARMA(p, q) model.
    Returns predictions (y_hat) and residuals (eps).
    """
    p = len(phi)
    q = len(theta)
    n = len(data)
    eps = np.zeros(n)
    y_hat = np.zeros(n)

    # For the first max(p, q) steps, we don't have enough history
    for t in range(max(p, q)):
        y_hat[t] = mu
        eps[t] = data[t] - y_hat[t]

    for t in range(max(p, q), n):
        # AR part: sum of phi_i * X_{t-i}
        ar_part = 0.0
        for i in range(p):
            ar_part = ar_part + phi[i] * data[t - i - 1]

        # MA part: sum of theta_j * eps_{t-j}
        ma_part = 0.0
        for j in range(q):
            ma_part = ma_part + theta[j] * eps[t - j - 1]

        y_hat[t] = mu + ar_part + ma_part
        eps[t] = data[t] - y_hat[t]

    return y_hat, eps


def arma_loss(params, data, p, q, mu, sigma):
    """
    Negative log-likelihood loss for ARMA(p, q).
    params: [phi_1, ..., phi_p, theta_1, ..., theta_q]
    """
    phi = params[:p]
    theta = params[p : p + q]

    y_hat, eps = arma_model(phi, theta, mu, data)

    n = len(data)
    loss = 0.0
    for t in range(n):
        loss = loss + np.log(sigma) + (eps[t] ** 2) / (2 * sigma**2)
    return loss


def arma_predict_one_step(phi, theta, mu, data):
    """
    One-step-ahead prediction on data using fitted ARMA parameters.
    For each t, predict X_t using actual past values and past residuals.
    """
    p = len(phi)
    q = len(theta)
    n = len(data)
    eps = np.zeros(n)
    y_hat = np.zeros(n)

    for t in range(max(p, q)):
        y_hat[t] = data[t]
        eps[t] = 0

    for t in range(max(p, q), n):
        ar_part = 0.0
        for i in range(p):
            ar_part = ar_part + phi[i] * data[t - i - 1]

        ma_part = 0.0
        for j in range(q):
            ma_part = ma_part + theta[j] * eps[t - j - 1]

        y_hat[t] = mu + ar_part + ma_part
        eps[t] = data[t] - y_hat[t]

    return y_hat


# --- Fit the ARMA model on training data ---

best_rmse = float("inf")
best_p = None
best_q = None
best_phi = None
best_theta = None
best_pred = None

mu = np.mean(training_data)
sigma = np.std(training_data)

for p in range(1, 5):
    for q in range(1, 5):
        init_params = np.zeros(p + q)
        try:
            result = minimize(
                arma_loss,
                init_params,
                args=(training_data, p, q, mu, sigma),
                method="Nelder-Mead",
                tol=1e-4,
            )
            phi = result.x[:p]
            theta = result.x[p : p + q]
            pred = arma_predict_one_step(phi, theta, mu, validation_data)
            rmse = np.sqrt(np.mean((validation_data - pred) ** 2))
            print(f"p={p}, q={q}, RMSE={rmse}")
            if rmse < best_rmse:
                best_rmse = rmse
                best_p = p
                best_q = q
                best_phi = phi
                best_theta = theta
                best_pred = pred
        except Exception as e:
            print(f"Failed for p={p}, q={q}: {e}")

print(f"\nBest (p, q): ({best_p}, {best_q}) with RMSE: {best_rmse}")
print("Best phi:", best_phi)
print("Best theta:", best_theta)

test_pred = arma_predict_one_step(best_phi, best_theta, mu, test_data)

# Plot the best result
plt.figure(figsize=(12, 5))
plt.plot(test_data, label="Test Data", color="blue")
plt.plot(
    test_pred, label=f"Best ARMA({best_p},{best_q}) Fit", color="red", linestyle="--"
)
plt.title(f"Test Data vs Best ARMA({best_p},{best_q}) Fit  |  RMSE = {best_rmse:.4f}")
plt.xlabel("Time step")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig("task2_arma_best_fit.png")
plt.show()


#### Plotting
# def acf(x, nlags):
#     x = list(x)
#     n = len(x)
#     mean = sum(x) / n
#     acf_vals = []
#     for k in range(nlags + 1):
#         numerator = 0.0
#         denominator = 0.0
#         for t in range(k, n):
#             numerator += (x[t] - mean) * (x[t - k] - mean)
#         for t in range(n):
#             denominator += (x[t] - mean) ** 2
#         acf_vals.append(numerator / denominator)
#     return acf_vals


# def pacf(x, nlags):
#     x = np.array(x)
#     pacf_vals = [1.0]
#     for k in range(1, nlags + 1):
#         X = []
#         y = []
#         for t in range(k, len(x)):
#             X.append([x[t - i - 1] for i in range(k)])
#             y.append(x[t])
#         X = np.array(X)
#         y = np.array(y)
#         if len(X) > 0:
#             coef = np.linalg.lstsq(X, y, rcond=None)[0]
#             pacf_vals.append(coef[-1])
#         else:
#             pacf_vals.append(0.0)
#     return pacf_vals


# nlags = 20
# acf_vals = acf(training_data, nlags)
# pacf_vals = pacf(training_data, nlags)

# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.stem(range(nlags + 1), acf_vals)
# plt.title("ACF (Training Data)")
# plt.xlabel("Lag")
# plt.ylabel("Autocorrelation")

# plt.subplot(1, 2, 2)
# plt.stem(range(nlags + 1), pacf_vals)
# plt.title("PACF (Training Data)")
# plt.xlabel("Lag")
# plt.ylabel("Partial Autocorrelation")

# plt.tight_layout()
# plt.savefig("acf_pacf_training.png")
# plt.show()
