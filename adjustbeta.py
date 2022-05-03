import numpy as np


def Hbeta(D, beta=1.0):
    """
    Compute entropy(H) and probability(P) from nxn distance matrix.

    Parameters
    ----------
    D : numpy.ndarray
        distance matrix (n,n)
    beta : float
        precision measure
    .. math:: \beta = \frac{1}/{(2 * \sigma^2)}

    Returns
    -------
    H : float
        entropy
    P : numpy.ndarray
        probability matrix (n,n)
    """
    num = np.exp(-D * beta)
    den = np.sum(np.exp(-D * beta), 0)
    P = num / den
    H = np.log(den) + beta * np.sum(D * num) / (den)
    return H, P


def adjustbeta(X, tol, perplexity, D):
    """
    Precision(beta) adjustment based on perplexity

    Parameters
    ----------
    X : numpy.ndarray
        data input array with dimension (n,d)
    tol : float
        tolerance for the stopping criteria of beta adjustment
    perplexity : float
        perplexity can be interpreted as a smooth measure of the effective number of neighbors

    Returns
    -------
    P : numpy.ndarray
        probability matrix (n,n)
    beta : numpy.ndarray
        precision array (n,1)
    """
    (n, d) = X.shape
    # Need to compute D here, which is nxn distance matrix of X
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1 : n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.0
                else:
                    beta[i] = (beta[i] + betamax) / 2.0
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.0
                else:
                    beta[i] = (beta[i] + betamin) / 2.0

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i + 1 : n]))] = thisP

    return P, beta
