# Parametric-Eigenvalues-and-Vectors-of-Transmission-Lines

This repository has a [jupyter notebook](code/unshuffled_eigen.ipynb) that compares some algorithms to obtain an unshuffled set of eigen values and vectors to allow a parametric analysis of them. It uses the telegrapher's equation for a multiconductor transmission line as contextual example.

Originally uploaded to Code Ocean. See [REPRODUCING.md](REPRODUCING.md).

Summary: try [eigenshuffle](https://github.com/ograsdijk/eigenshuffle). If that does not work,
then use the Levenberg--Marquardt Method (see the code below).

```python
import warnings
import numpy as np
from scipy.constants import mu_0, epsilon_0, pi
from scipy.optimize import least_squares

def eig_levenberg_marquardt(S_omega, tol=1e-8, max_iter=1000):
    """Solve the complex symmetric eigenvalue problem for multiple frequencies.

    S_omega may need to be normalized. Example:
    ```
    A = np.load("YZ.npy")
    K, N, _ = A.shape
    fhz = np.load("frequency_Hz.npy")
    omega = 2 * np.pi * fhz
    
    # normalization
    w2u0e0 = -omega**2 * mu_0 * epsilon_0
    S_omega = np.array([A[i] / (-w2u0e0[i]) - np.identity(N) for i in range(K)])
    
    eigvals, eigvecs = eigen_levenberg_marquardt(S_omega, tol=1e-11, max_iter=10000)

    # scale back the eigenvalues; eigenvectors remain unchanged
    gamma = np.array([np.sqrt(-w2u0e0 * (1 + eigvals[:, i])) for i in range(N)]).T
    ```
    
    Parameters
    ----------
    S_omega: K x N x N array of complex matrices S(ω) for K frequencies
    tol: Tolerance for convergence
    max_iter: Maximum number of iterations
    
    Returns
    -------
    eigenvalues: K x N array of complex eigenvalues
    eigenvectors: K x N x N array of complex eigenvectors (columns are eigenvectors)

    
    References
    ----------
    A. I. Chrysochos, T. A. Papadopoulos and G. K. Papagiannis, "Robust Calculation
    of Frequency-Dependent Transmission-Line Transformation Matrices Using the
    Levenberg–Marquardt Method," in IEEE Transactions on Power Delivery, vol. 29,
    no. 4, pp. 1621-1629, Aug. 2014, doi: 10.1109/TPWRD.2013.2284504.
    """
    K, N, _ = S_omega.shape
    eigenvalues = np.zeros((K, N), dtype=complex)
    eigenvectors = np.zeros((K, N, N), dtype=complex)
    
    # First solve the first frequency with standard eigensolver
    S0 = S_omega[0]
    eigvals, eigvecs = np.linalg.eig(S0)
    eigenvalues[0] = eigvals
    eigenvectors[0] = eigvecs
    
    # Prepare real-valued formulation for subsequent frequencies
    for k in range(1, K):
        S = S_omega[k]
        S_re = np.real(S)
        S_im = np.imag(S)
        
        # Previous solution as initial guess
        prev_eigvecs = eigenvectors[k-1]
        prev_eigvals = eigenvalues[k-1]
        
        # Solve for each eigenvalue/eigenvector pair
        for i in range(N):
            # Initial guess from previous frequency
            t_prev = prev_eigvecs[:, i]
            lambda_prev = prev_eigvals[i]
            
            # Real-valued formulation
            def residuals(x):
                t_re = x[:N]
                t_im = x[N:2*N]
                lambda_re = x[2*N]
                lambda_im = x[2*N+1]
                
                # Eigen equation residuals
                res1 = (S_re @ t_re - S_im @ t_im) - (lambda_re * t_re - lambda_im * t_im)
                res2 = (S_im @ t_re + S_re @ t_im) - (lambda_im * t_re + lambda_re * t_im)
                
                # Normalization constraints
                norm1 = np.sum(t_re**2) - np.sum(t_im**2) - 1  # |t|^2 = 1
                norm2 = 2 * np.sum(t_re * t_im)  # Ensures proper phase
                
                return np.concatenate([res1, res2, [norm1, norm2]])
            
            # Initial guess
            x0 = np.concatenate([
                np.real(t_prev),
                np.imag(t_prev),
                [np.real(lambda_prev), np.imag(lambda_prev)]
            ])
            
            # Normalize initial guess to satisfy constraints
            t_re, t_im = x0[:N], x0[N:2*N]
            denom = np.sqrt(np.sum(t_re**2) + np.sum(t_im**2))
            x0[:2*N] /= denom
            
            # Solve with Levenberg-Marquardt
            res = least_squares(
                residuals,
                x0,
                method='lm',
                xtol=tol,
                ftol=tol,
                max_nfev=max_iter
            )
            
            if not res.success:
                msg = f"Warning: Did not converge for frequency {k}, eigenvalue {i}\nResidual norm: {np.linalg.norm(res.fun)}"
                warnings.warn(msg)
            
            # Extract solution
            t_re = res.x[:N]
            t_im = res.x[N:2*N]
            lambda_re = res.x[2*N]
            lambda_im = res.x[2*N+1]
            
            # Store solution
            eigenvectors[k, :, i] = t_re + 1j * t_im
            eigenvalues[k, i] = lambda_re + 1j * lambda_im
            
            # Ensure eigenvectors are properly normalized
            eigvec = eigenvectors[k, :, i]
            eigenvectors[k, :, i] = eigvec / np.sqrt(np.sum(np.abs(eigvec)**2))
    
    return eigenvalues, eigenvectors
```