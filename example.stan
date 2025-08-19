data {
    int<lower=1> n;               // number of observations
  vector[n] l;                    // covariate
  int<lower=0,upper=1> a[n];      // treatment indicator
  real y[n];                      // observed outcome
}

/*transformed data{
    // if you want to specify a specific value, 
    // uncomment, comment out rho declaration in parameters block
    // comment out rho prior in model block.
    real<lower=-1, upper=1> rho=0 ;
}
*/

parameters {
  // outcome regression parameters
  real beta01;
  real beta11;
  real beta00;
  real beta10;
  
  // Full covariance matrix
  real<lower=0> sigma1;
  real<lower=0> sigma0;
  real<lower=-1,upper=1> rho;

  // unobserved potential outcomes
  vector[n] y_m;  
  
  // bayesian bootstrap parameters
  simplex[n] theta_l;  
  real eta;
  real<lower=0> tau;

}

transformed parameters {
  
  // outcome mean vector
  vector[n] mu1 = beta01 + beta11 * l;
  vector[n] mu0 = beta00 + beta10 * l;

  // Full covariance matrix
  matrix[2,2] Sigma;
  Sigma[1,1] = square(sigma1);
  Sigma[2,2] = square(sigma0);
  Sigma[1,2] = rho * sigma1 * sigma0;
  Sigma[2,1] = Sigma[1,2];
}


model {
  // prior on Cor( Y(1) , Y(0) )
  // improper \propto 1 priors on other pars
  rho ~ uniform(-.9, .9); // non-identifiable - divergent iterations expected.
  
  // covariate model contribution
  l ~ normal(eta, tau);

  // outcome model contribution
  for (i in 1:n) {
    vector[2] mu_i = [ mu1[i] , mu0[i] ]' ;
    vector[2] y_c; // complete potential outcome - observed with missing

    if (a[i] == 1) {
      y_c = [ y[i] , y_m[i] ]' ;
    } else {
      y_c = [ y_m[i] , y[i] ]' ;    
    }

    y_c ~ multi_normal(mu_i, Sigma);
  }
  
}


generated quantities{
  int S=1000;
  real ITEs[n];
  real SATE;
  
  vector[n] CATE;
  
  real PATE;
  real PATE_bb;
  real PATE_ecdf;
  
  simplex[n] phi_L = dirichlet_rng(rep_vector(1, n));
  
  // compute the ITE and SATE
  for( i in 1:n){
    if (a[i] == 1) {
      ITEs[i]  = y[i] - y_m[i];       
    } else {
      ITEs[i]  = y_m[i] - y[i];       
    }
  }
  
  SATE = mean(ITEs);
  
  
  // compute the CATE and PATE
  CATE = (beta01 + beta11 * l ) - ( beta00 + beta10 * l );
  
  // PATE - Monte Carlo simulation from parametric model
  PATE = 0;
  for (s in 1:S){
    real l_sim = normal_rng(eta, tau);
    PATE += ( (beta01 + beta11 * l_sim ) - ( beta00 + beta10 * l_sim ) );
  }
  PATE = PATE/S;
  
  // PATE - Bayesian Bootstrap
  PATE_bb = 0;
  for (i in 1:n){
    PATE_bb += ( (beta01 + beta11 * l[i] ) - ( beta00 + beta10 * l[i] ) )*phi_L[i];
  }
  
  // PATE - Empirical distribution
  PATE_ecdf = 0;
  for (i in 1:n){
    PATE_ecdf += ( (beta01 + beta11 * l[i] ) - ( beta00 + beta10 * l[i] ) )*(1.0/n);
  }

}
