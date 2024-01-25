import numpy as np


def gaussian(X, mu, sigma):
    """Compute the probability density of a Gaussian distribution"""
    d = X.shape[1]
    det = np.linalg.det(sigma)
    inv = np.linalg.inv(sigma)
    const = 1.0 / ((2 * np.pi) ** (d / 2) * np.sqrt(det))
    x_mu = X - mu
    return const * np.exp(-0.5 * np.sum(np.dot(x_mu, inv) * x_mu, axis=1))


def Expectation(X, k, pi, mu, sigma):
    """Expectation step: calculate the membership weights for each data point"""
    n = X.shape[0]
    w = np.zeros((n, k))
    for j in range(k):
        w[:, j] = pi[j] * gaussian(X, mu[j], sigma[j])
    w = w / np.sum(w, axis=1, keepdims=True)
    return w


def MaximizeMean(X, k, w):
    """Maximization step: calculate the new maximum likelihood means for each Gaussian"""
    mu = np.zeros((k, X.shape[1]))
    for j in range(k):
        mu[j] = np.sum(w[:, j][:, np.newaxis] * X, axis=0) / np.sum(w[:, j])
    return mu


def MaximizeCovariance(X, k, w, mu):
    """Maximization step: calculate the new maximum likelihood covariances for each Gaussian"""
    d = X.shape[1]
    sigma = np.zeros((k, d, d))
    for j in range(k):
        x_mu = X - mu[j]
        sigma[j] = np.dot((w[:, j][:, np.newaxis] * x_mu).T, x_mu * w[:, j][:, np.newaxis]) / np.sum(w[:, j])
    return sigma


def MaximizeMixtures(k, w):
    """Maximization step: calculate the new maximum likelihood mixture weights for each Gaussian"""
    pi = np.mean(w, axis=0)
    return pi


def EM(X, k, pi0, mu0, sigma0, nIter):
    """Run the EM algorithm for nIter steps and return the parameters of the underlying GMM"""
    pi = pi0
    mu = mu0
    sigma = sigma0
    log_likelihoods = []
    for i in range(nIter):
        # Expectation step
        w = Expectation(X, k, pi, mu, sigma)

        # Maximization step
        mu = MaximizeMean(X, k, w)
        sigma = MaximizeCovariance(X, k, w, mu)
        pi = MaximizeMixtures(k, w)

        # Compute log likelihood
        log_likelihood = np.sum(np.log(np.sum(pi[j] * gaussian(X, mu[j], sigma[j]) for j in range(k))))
        log_likelihoods.append(log_likelihood)

        # Check for convergence
        if i > 0 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < 1e-4:
            break

    return pi, mu, sigma


with open('/Users/naveenverma/Desktop/NewStart/Dataset/a4-data/a4-q5-data/multigauss.txt', 'r') as f:
    # Skip the header lines
    for _ in range(3):
        next(f)
    # Read the data
    data = np.loadtxt(f)

# Set number of clusters
k = 3

# Initialize mixture weights, means, and covariances
pi0 = np.ones(k) / k
mu0 = data[np.random.choice(data.shape[0], size=k, replace=False)]
sigma0 = np.array([np.eye(data.shape[1]) for j in range(k)])

pi, mu, sigma = EM(data, k, pi0, mu0, sigma0, nIter=100)
print("pi :",pi)
print("mu :", mu)
print("sigma :", sigma)


