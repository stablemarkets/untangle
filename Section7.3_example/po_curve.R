
set.seed(123)
library(MASS)
library(rstan)
library(latex2exp)

## ---------- helper: AR(1)-type covariance matrix ----------
make_sigma <- function(K, sigma2 = 1, rho = 0.5) {
  mat <- outer(1:K, 1:K, function(i, j) rho^abs(i - j))
  sigma2 * mat
}

## ---------- main simulation function ----------
sim_data <- function(n = 1000,
                     K = 10,
                     alpha0 = rep(0, K - 1),      # intercepts for multinomial logits (2..K)
                     alpha1 = rep(0.5, K - 1),    # slope on L for logits (2..K)
                     beta0 = seq(-1, 1, length.out = 10),  # mean intercepts for Y(a)
                     beta1 = rep(0.3, 10),        # slope of Y(a) means on L
                     sigma2 = 1,
                     rho = 0.5) {
  
  if (length(beta0) != K || length(beta1) != K)
    stop("beta0 and beta1 must be length K")
  
  ## 1. Confounder
  L <- rnorm(n, mean = 0, sd = 1)
  
  ## 2. Treatment from multinomial with logits depending on L
  ##    Category 1 is baseline; for k = 2..K:
  ##    log( P(A=k)/P(A=1) ) = alpha0[k-1] + alpha1[k-1] * L
  linpred <- sapply(2:K, function(k) alpha0[k-1] + alpha1[k-1] * L)  # n x (K-1)
  exp_lp  <- exp(linpred)
  denom   <- 1 + rowSums(exp_lp)
  
  P <- matrix(NA, nrow = n, ncol = K)
  P[, 1]      <- 1 / denom
  P[, 2:K]    <- exp_lp / denom
  
  colnames(P) <- paste0("A", 1:K)
  
  ## Draw A as categorical
  A <- apply(P, 1, function(prob) sample.int(K, size = 1, prob = prob))
  
  ## 3. Potential outcomes Y(1),...,Y(K) ~ MVN(mean_i, Sigma)
  ## mean_i(a) = beta0[a] + beta1[a] * L_i
  Sigma <- make_sigma(K = K, sigma2 = sigma2, rho = rho)
  MU <- cbind(sapply(1:K, function(a) beta0[a] + beta1[a] * L))
  
  Y_pot <- t(apply(MU, 1, function(mu_i) {
    MASS::mvrnorm(n = 1, mu = mu_i, Sigma = Sigma)
  }))
  colnames(Y_pot) <- paste0("Y", 1:K)
  
  ## Observed outcome: Y_obs = Y_{A}
  Y_obs <- Y_pot[cbind(1:n, A)]
  
  ## Return everything
  return(list(data = data.frame(L=L,A= A,Y=Y_obs), Y_pot = Y_pot))
}

## ---------- example usage ----------
n=200
sim <- sim_data(n = n)

stan_data = list(n=n, y_obs=sim$data$Y, A=sim$data$A, L=sim$data$L, K=10)

mod = stan_model("po_curve.stan")

run_stan_mod = function(rho){
  
  stan_res = sampling(mod, 
                      data = c( stan_data , rho = rho), 
                      chains=1, warmup=1000, iter=2000, seed=123)
  
  Y_full = extract(stan_res, 'Y_full')$Y_full
  cond_mean_y1 = extract(stan_res, 'cond_mean_y_1')$cond_mean_y_1
  
  post_mean_yfull = colMeans(Y_full[,1,])
  lwr_yfull = apply(Y_full[,1,],2, quantile, probs=.025)
  upr_yfull = apply(Y_full[,1,],2, quantile, probs=.975)
  
  post_mean_y1 = colMeans(cond_mean_y1)
  lwr_y1 = apply(cond_mean_y1,2, quantile, probs=.025)
  upr_y1 = apply(cond_mean_y1,2, quantile, probs=.975)
  
  return(list(post_mean_yfull=post_mean_yfull,
              lwr_yfull=lwr_yfull, upr_yfull=upr_yfull,
              post_mean_y1=post_mean_y1,
              lwr_y1=lwr_y1, upr_y1=upr_y1) )
}

res_s1 = run_stan_mod(rho = .9)
res_s2 = run_stan_mod(rho = 0)

png('po_curve.png', width=1000, height=500)
par(mfrow=c(1,2))

plot(1:10, sim$Y_pot[1,], pch=20, type='o',ylim=c(-3,3), col='pink', 
     main = TeX('Posterior Inference for $\\bar{Y}_1$, $\\rho=.9$'),
     xlab= 'Treatment level, a=1,2,...,10', 
     ylab = 'Y')
segments(1:10, res_s1$lwr_yfull, 1:10, res_s1$upr_yfull, col='black')
points(1:10, res_s1$post_mean_yfull, pch=20, col='black')

points(1:10+.25, res_s1$post_mean_y1, col='blue', pch=20)
segments(1:10+.25, res_s1$lwr_y1, 1:10+.25, res_s1$upr_y1, col='blue')

points(1:10, sim$Y_pot[1,], pch=20, type='o',ylim=c(-5,5), col='pink', lwd=2)
points(sim$data$A[1], sim$data$Y[1], col='red', cex=2, pch=20)

legend('topleft', 
       legend=c(TeX('True $Y_1(a)$'),
                TeX('Posterior of $Y_1(a)$'),
                TeX('Posterior of $E[Y(a) | L=l_1]$')), 
       pch=c(20,20,20), lty=c(1,1,1), 
       col=c('red','black', 'blue'), 
       bty='n')


plot(1:10, sim$Y_pot[1,], pch=20, type='o',ylim=c(-3,3), col='pink', 
     main = TeX('Posterior Inference for $\\bar{Y}_1$, $\\rho=0$'),
     xlab= 'Treatment level, a=1,2,...,10', 
     ylab = 'Y')
segments(1:10, res_s2$lwr_yfull, 1:10, res_s2$upr_yfull, col='black')
points(1:10, res_s2$post_mean_yfull, pch=20, col='black')

points(1:10 +.25, res_s2$post_mean_y1, col='blue', pch=20)
segments(1:10 +.25, res_s2$lwr_y1, 1:10+.25, res_s2$upr_y1, col='blue')

points(1:10, sim$Y_pot[1,], pch=20, type='o',ylim=c(-5,5), col='pink', lwd=2)
points(sim$data$A[1], sim$data$Y[1], col='red', cex=2, pch=20)
dev.off()

res_s2$upr_yfull - res_s2$lwr_yfull
res_s1$upr_yfull - res_s1$lwr_yfull

