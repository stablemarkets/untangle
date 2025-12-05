data {
  int<lower=1> n;                       // number of subjects
  int<lower=1> K;                       // number of treatment levels
  vector[n] L;                          // confounder L_i
  int<lower=1, upper=K> A[n];           // observed treatment A_i
  vector[n] y_obs;                      // observed outcome Y_i(A_i)
  real<lower=0, upper=1> rho;           // correlation parameter
}

parameters {
  // mean structure: mu_ik = beta0[k] + beta1[k] * L_i
  vector[K] beta0;
  vector[K] beta1;

  // AR(1)-type covariance: Sigma_kl = sigma2 * rho^{|k-l|}
  real<lower=0> sigma2;                 // variance scale
  //real<lower=.25, upper=.75> rho;          // correlation parameter

  // K-1 missing potential outcomes per subject
  // row i: the K-1 unobserved components of Y_i(1:K)
  matrix[n, K - 1] y_mis;
}

transformed parameters {
  // full potential-outcome matrix: row i = (Y_i(1),...,Y_i(K))
  matrix[n, K] Y;

  // covariance and its Cholesky factor
  matrix[K, K] Sigma;
  matrix[K, K] L_Sigma;

  // build AR(1) covariance Sigma
  for (k in 1:K) {
    for (j in 1:K) {
      int d = abs(k - j);               // integer distance |k - j|
      Sigma[k, j] = sigma2 * pow(rho, d);
    }
  }

  // for (k in 1:K) {
  //   for (j in 1:K) {
  //     if(j==k){
  //       Sigma[k, j] = sigma2;
  //     }else{
  //       Sigma[k, j] = 0;
  //     }
  // 
  //   }
  // }

  L_Sigma = cholesky_decompose(Sigma);

  // assemble full Y_i(1:K) from y_obs and y_mis
  for (i in 1:n) {
    int m = 1;                          // index within the K-1 missing columns
    for (k in 1:K) {
      if (k == A[i]) {
        // observed component
        Y[i, k] = y_obs[i];
      } else {
        // missing component (parameter)
        Y[i, k] = y_mis[i, m];
        m += 1;
      }
    }
  }
}

model {
  // weakly informative priors
  beta0  ~ normal(0, 5);
  beta1  ~ normal(0, 5);
  sigma2 ~ normal(0, 2);                // half-normal due to lower=0
  //rho    ~ uniform(.25, .75);

  // likelihood: for each i, Y_i(1:K) ~ MVN_K(mu_i, Sigma)
  for (i in 1:n) {
    vector[K] mu_i;
    vector[K] y_i;

    // build mean and outcome vectors
    for (k in 1:K) {
      mu_i[k] = beta0[k] + beta1[k] * L[i];
      y_i[k]  = Y[i, k];
    }

    y_i ~ multi_normal_cholesky(mu_i, L_Sigma);
  }
}

generated quantities {
  // (posterior draws of Y_full include both observed and imputed).
  matrix[n, K] Y_full;
  vector[K] cond_mean_y_1;
  vector[K] cond_mean_y_2;

  Y_full = Y;
  
  cond_mean_y_1 = beta0 + beta1 * L[1];
  cond_mean_y_2 = beta0 + beta1 * L[15];
  
  
}
