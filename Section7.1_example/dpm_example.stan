data{
  int<lower=0> K; 
  int<lower=0> n; 
  int<lower=0> n_1; 
  int<lower=0> n_0; 
  
  // grid of values for computing regression line
  int<lower=0> N_test;
  real l_test[N_test];
  
  // observed data for those with/without missing outcomes
  real l_1[n_1];
  real y_1[n_1];
  
  real l_0[n_0];
  real y_0[n_0];
  
}

transformed data{
    real rho=0;
}

parameters {
  real<lower=0> ax;
  real<lower=0> bx;
  real cx;
  real<lower=0> dx;
  real<lower=0> a;
  real<lower=0> b;
  
  real b0m;
  real b1m;
  real b2m;
  
  real<lower=0> sigma0;
  real<lower=0> sigma1;
  real<lower=0> sigma2;
  
  vector[K] mu_x;
  vector<lower=0>[K] phi_x;
  
  vector[K] beta00;
  vector[K] beta10;

  vector[K] beta01;
  vector[K] beta11;
  
  vector<lower=0>[K] tau_mix0;
  vector<lower=0>[K] tau_mix1;
  real <lower=0,upper=1> q[K];
  real <lower=0> alpha;
  
  real y_1m[n_0];
  real y_0m[n_1];
  
  real xi1;
  real xi2;
  
}

transformed parameters{
  simplex[K] w;
  
  w[1] = q[1];
  for(j in 2:(K-1) ){
    w[j] = q[j]*(1-q[j-1])*w[j-1]/q[j-1];
  }
  w[K] = 1 - sum(w[1:(K-1)]);
  
}

model {
  alpha ~ gamma(1,1);
  
  // runs a bit faster and better results with these hyperpriors
  a ~ normal(0,1);
  b ~ normal(0,1);
  
  ax ~ normal(0,1);
  bx ~ normal(0,1);
  
  b0m ~ normal(0,1);
  b1m ~ normal(0,1);
  b2m ~ normal(0,1);
  
  cx ~ normal(0,1);
  dx ~ normal(0,1);
  
  sigma2 ~ normal(0,1);
  sigma1 ~ normal(0,1);
  sigma0 ~ normal(0,1);
  
  xi1 ~ normal(0,1);
  xi2 ~ normal(0,1);
  
  for(j in 1:K){
    q[j] ~ beta(1, alpha);
    
    mu_x[j] ~ normal(cx,dx);
    phi_x[j] ~ normal(ax,bx);
    
    beta10[j] ~ normal( b1m, sigma1 );
    beta00[j] ~ normal( b0m, sigma0 );
    
    beta01[j] ~ normal( b1m, sigma1 );
    beta11[j] ~ normal( b0m, sigma0 );
    
    tau_mix0[j] ~ inv_gamma ( a, b );
    tau_mix1[j] ~ inv_gamma ( a, b );
  }
  
  for(i in 1:n_1){
    vector[K] lps = log(w);
    for(k in 1:K){
      lps[k] += normal_lpdf(y_1[i] |  beta01[k] + beta11[k]*l_1[i] + (rho*(tau_mix1[k]/tau_mix0[k]))*(y_0m[i] -(beta00[k] + beta10[k]*l_1[i]) ), tau_mix1[k]*(1-rho*rho));
      lps[k] += normal_lpdf(y_0m[i] | beta00[k] + beta10[k]*l_1[i], tau_mix0[k]);
      lps[k] += normal_lpdf(l_1[i] | mu_x[k], phi_x[k]);
    }
    target += log_sum_exp(lps);
  }
  
  for(i in 1:n_0){
    vector[K] lps = log(w);
    for(k in 1:K){
      lps[k] += normal_lpdf(y_1m[i] |  beta01[k] + beta11[k]*l_0[i] + (rho*(tau_mix1[k]/tau_mix0[k]))*(y_0[i] -(beta00[k] + beta10[k]*l_0[i]) ), tau_mix1[k]*(1-rho*rho));
      lps[k] += normal_lpdf(y_0[i] | beta00[k] + beta10[k]*l_0[i], tau_mix0[k]);
      lps[k] += normal_lpdf(l_0[i] | mu_x[k], phi_x[k]);
    }
    target += log_sum_exp(lps);
  }
  
  
}

generated quantities{
  
  real cond_mean_y_a1[N_test];
  real cond_mean_y_a0[N_test];
  real cond_density_l[N_test];
  
  // E[Y | A=1, L=l]
  for( i in 1:N_test){
    vector[K] wght;
    vector[K] cond_mod;
    for(k in 1:K){
      wght[k] = w[k]*exp(normal_lpdf(l_test[i] | mu_x[k], phi_x[k] ) );
      cond_mod[k] = (beta01[k] + beta11[k]*l_test[i])*wght[k];
    }
    cond_mean_y_a1[i] = sum(cond_mod) / sum(wght) ;
  }
  
  // E[Y | A=0, L=l]
  for( i in 1:N_test){
    vector[K] wght;
    vector[K] cond_mod;
    for(k in 1:K){
      wght[k] = w[k]*exp(normal_lpdf(l_test[i] | mu_x[k], phi_x[k] ) );
      cond_mod[k] = (beta00[k] + beta10[k]*l_test[i])*wght[k];
    }
    cond_mean_y_a0[i] = sum(cond_mod) / sum(wght) ;
  }
  
  for( i in 1:N_test){
    vector[K] cond_den;
    for(k in 1:K){
      cond_den[k] =exp(normal_lpdf(l_test[i] | mu_x[k], phi_x[k] ) )*w[k];
    }
    cond_density_l[i] = sum(cond_den);
  }
  
}
