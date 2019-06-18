import numpy as np
from scipy.linalg import inv, cholesky

def sqrt_kalman_smoother(z, A, C, Q, R, X1, P1, N):
    # Get the dimensions
    n = X1.shape[0]
    m = z.shape[0]

    x_prior = np.zeros((N+1, n))
    x_prior[0] = X1
    x = np.zeros((N,n))

    P_sqrt_prior = np.zeros((N+1, n, n))
    P_sqrt_prior[0] = cholesky(P1)
    P_sqrt = np.zeros(N,n,n)

    Q_sqrt = cholesky(Q)
    R_sqrt = cholesky(R)

    R1 = np.zeros((n+m,n+m))
    R2 = np.zeros((n+m,n+m))
    R3 = np.zeros((n*2,n))
    R4 = np.zeros((n*2,n))

    K = np.zeros((N,n,m))
    z_prior = np.zeros(N,m) 
    z_postier = np.zeros(N,m)
    residual = np.zeros(N,m)
    error = np.zeros(N,m)

    # Kalman filter
    for t in range(N):
        # --- UPDATE PHASE ---
        # Covariance update
        #
        # qr [R_sqrt,       0  ] = [S.T, (K * S.T).T   ]
        #    [P_sqrt_prior, C.T]   [0,   P_sqrt_postier]
        R1[:m,:m] = R_sqrt
        R1[m:,:m] = P_sqrt_prior @ C.T
        R1[m:,m:] = P_sqrt_prior

        R2 = np.qr(R1)

        K[t] = R2[:m,m:].T @ inv(R2[:m,:m])
        P_sqrt[t] = R2[m:,m:]

        # Means update
        z_prior[t] = C @ x_prior[t]
        residual[t] = z[t] - z_prior[t]
        x[t] = x_prior[t] + K[t] @ residual[t]

        # error
        z_postier[t] = C @ x[t]
        error[t] = z[t] = z_postier[t]

        # --- PREDICT PHASE ---
        # Means predict
        x_prior[t+1] = A @ x[t]

        # Covariance predict
        # 
        # qr [P_sqrt_previous * A.T] = [P_sqrt_prior]
        #    [Q_sqrt               ]   [0           ]
        R3[:n,:] = P_sqrt[t] @ A.T
        R3[n:,:] = Q_sqrt

        R4 = np.qr(R3)

        P_sqrt_prior[t+1] = R4[:n,:]

    # Smooth
    # Initialization
    x_smooth = np.zeros((N+1,n))
    x_smooth[N] = x_prior[N].copy()
    x_smooth[N-1] = x[N-1].copy()

    P_sqrt_smooth = np.zeros((N+1,n,n))
    P_sqrt_smooth[N] = P_sqrt_prior[N].copy()
    P_sqrt_smooth[N-1] = P_sqrt[N-1].copy()

    M_smooth = np.zeros(N,n,n)
    M_smooth[N-1] = P_sqrt[N].T @ P_sqrt[N] @ A.T

    z_smooth = np.zeros((N,m))
    error_smooth = np.zeros((N,m))

    R5 = np.zeros((3*n,2*n))
    R6 = np.zeros((3*n,2*n))

    for t in range(N-2,-1,-1):
        P = P_sqrt[t] @ P_sqrt[t].T
        P_prior = P_sqrt_prior[t+1] @ P_sqrt_prior[t+1].T
        AP = A @ P
        Jt = AP.T @ inv(P_prior)

        R5[:n,:n] = P_sqrt[t] @ A.T
        R5[:n,n:] = P_sqrt[t]
        R5[n:2*n,:n] = Q_sqrt
        R5[2*n:,n:] = P_sqrt_smooth[t+1] @ Jt.T

        R6 = np.qr(R5)

        P_sqrt_smooth[t] = R6[n:2*n,n:]

        x_smooth[t] = x[t] + Jt @ (x_smooth[t+1] - x_prior[t+1])
        z_smooth[t] = C @ x_smooth[t]
        error_smooth[t] = z[t] - z_smooth[t]

        if t == N-2:
            M_smooth[t] = (np.eye(n) - K[N-1] @ C) @ AP
        else:
            M_smooth[t] = (P_sqrt[t+1].T @ P_sqrt[t+1] + Jtp1 @ APp1) @ Jt.T
        Jtp1 = Jt
        APp1 = M_smooth[t] - AP
        
    # Calculate the H matrix
    H_xx = np.zeros((n,n))
    H_xx1 = np.zeros((n,n))
    H_x1x1 = np.zeros((n,n))
    for t in range(N):
        H_xx += x_smooth[t] @ x_smooth[t].T + P_sqrt_smooth[t].T @ P_sqrt_smooth[t]
        H_xx1 += x_smooth[t] @ x_smooth[t+1].T + M_smooth[t]
        H_x1x1 += x_smooth[t+1] @ x_smooth[t+1].T + P_sqrt_smooth[t+1] @ P_sqrt_smooth[t+1].T