from scipy.linalg import inv

def sqrt_kalman_smoother():
    x_prior[0] = X1
    P_sqrt_prior[0] = P1_sqrt

    # Get the dimensions
    n = X1.shape[0]
    m = z.shape[0]

    # Kalman filter
    for t in range(N):
        # --- UPDATE PHASE ---
        # Covariance update
        # R1 = np.zeros((n+m, n+m))
        # qr [R_sqrt,       0  ] = [S.T, (K * S.T).T   ]
        #    [P_sqrt_prior, C.T]   [0,   P_sqrt_postier]
        R1[:m,:m] = R_sqrt
        R1[:m,m:] = np.zeros((m,n))
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
    x_smooth[N] = x_prior[N].copy()
    x_smooth[N-1] = x[N-1].copy()
    P_sqrt_smooth[N] = P_sqrt_prior[N].copy()
    P_sqrt_smooth[N-1] = P_sqrt[N-1].copy()
    M_smooth[N-1] = P_sqrt[N].T @ P_sqrt[N] @ A.T

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
    for i in range(N):
        H_xx += x_smooth[t] @ x_smooth[t].T + P_sqrt_smooth[t].T @ P_sqrt_smooth[t]
        H_xx1 += x_smooth[t] @ x_smooth[t+1].T + M_smooth[t]
        H_x1x1 += x_smooth[t+1] @ x_smooth[t+1].T + P_sqrt_smooth[t+1] @ P_sqrt_smooth[t+1].T
