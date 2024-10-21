# EM_Function

#print(len(range(num_iter)), len(sigma_seas_values), len(sigma_eps_values))
num_iter = 100
# Some arbitrary noise values for now
sigma_trend = 0.01
sigma_seas = 1
sigma_eps = 1
Q = np.array([[sigma_trend**2, 0.], [0., sigma_seas**2]])  # Process noise covariance matrix
y_train=y[:n]

# Initialize the list to store sigma_seas values
sigma_seas_values = [sigma_seas]
sigma_eps_values = [sigma_eps]

for r in range(1,num_iter):
    
    # E-step
    
    H=sigma_eps**2
    model = LGSS(T, R, Q, Z, H, a1, P1)
    kalman_f = kalman_filter(y_train, model)
    kalman_s= kalman_smoother(y_train, model,  kalman_f )
  
   
    # M-step
    
    #Update Q value/sigma_seas
    Q_update = np.mean([np.dot(kalman_s.eta_hat[:,:,i], kalman_s.eta_hat.T[i,:,:]) + kalman_s.eta_cov[:,:,i] for i in range(n)], axis=0)
    Q[1,1] = (Q_update[1,1])

     #Update sigma_eps 
    sigma2_eps_new =(np.mean(kalman_s.eps_hat**2 + kalman_s.eps_var))
    sigma_eps=np.sqrt(sigma2_eps_new)
    
    #sigma_seas & sigma_eps 
    sigma_seas_values.append(Q[1][1])
    sigma_eps_values.append(sigma2_eps_new)