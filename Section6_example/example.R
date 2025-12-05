library(LaplacesDemon)
library(rstan)

simulate_data = function(n){
  l = rnorm(n) ## simulate covariate
  a = 1*( runif(n) < plogis(1 + 2*l) )  ## simulate treatment
  
  ## simulate potential outcomes
  mu = cbind(10 -4*l, 5 + 5*l)
  Sigma = matrix(c(1,0,0,1), nrow=2)
  po_mat = matrix(NA, nrow=n, ncol=2)
  
  for(i in 1:n){
    po_mat[i,]=rmvn(n=1, mu= mu[i,], Sigma= Sigma)  
  }
  
  ## grab observed outcome
  y = ifelse(a==1, po_mat[,1], po_mat[,2])
  
  ## output data
  data = data.frame(y, a, l, ITE = po_mat[,1] - po_mat[,2])
  return(data)
}

set.seed(1293)

### simulate some data
n = 50
data = simulate_data(n)

### run Stan model
stan_data = list(n=n, y=data$y, a=data$a, l=data$l)

mod = stan_model("example.stan")

stan_res = sampling(mod, data=stan_data, 
                    chains=1, warmup=5000, iter=10000,
                    seed=123)

### extract posterior draws of estimands
ITE_draws = extract(stan_res, 'ITEs')$ITEs
SATE_draws = extract(stan_res, 'SATE')$SATE
CATE_draws = extract(stan_res, 'CATE')$CATE

PATE_draws = extract(stan_res, 'PATE_exact')$PATE_exact
PATE_bb_draws = extract(stan_res, 'PATE_bb')$PATE_bb
PATE_ecdf_draws = extract(stan_res, 'PATE_ecdf')$PATE_ecdf

### compute subject-level mean and credible intervals
ITE_mean = colMeans(ITE_draws)
ITE_lwr = apply(ITE_draws,2,quantile, p=.025)
ITE_upr = apply(ITE_draws,2,quantile, p=.975)

CATE_mean = colMeans(CATE_draws)
CATE_lwr = apply(CATE_draws,2,quantile, p=.025)
CATE_upr = apply(CATE_draws,2,quantile, p=.975)


### plot results

png('comparison_fig.png',width=700, height=400)
par(mfrow=c(1,2))
np = 30
pd = rbind( data.frame(draws=SATE_draws, id='SATE'), 
            data.frame(draws=PATE_draws, id='PATE') )

boxplot(pd$draws ~ pd$id, 
        ylab = 'Posterior Draws', 
        xlab = 'Estimand',
        main='Posterior Draws of PATE and SATE')

plot(ITE_mean[order(ITE_lwr)][1:np], 1:np, xlim=c(-15,10), pch=20, 
     xlab = 'Posterior Draws', 
     ylab = 'Subject ID',
     main='Posterior Draws of ITEs and CATEs')
segments(ITE_lwr[order(ITE_lwr)][1:np], 1:np, 
         ITE_upr[order(ITE_lwr)][1:np], 1:np )

points(CATE_mean[order(ITE_lwr)][1:np], 1:np + .25, pch=20, col='red')
segments(CATE_lwr[order(ITE_lwr)][1:np], 1:np + .25, 
         CATE_upr[order(ITE_lwr)][1:np], 1:np + .25, col='red')

legend('topleft', legend = c('ITE', 'CATE'), 
       col=c('black','red'), lty=c(1,1))

dev.off()


png('bb_comparison.png',width=800, height=300)
par(mfrow=c(1,3))
hist(PATE_draws, xlim=c(-2,13), breaks=20, main='Exact',
     xlab='Posterior Draws')
hist(PATE_bb_draws, xlim=c(-2,13), breaks=20, main='Bayesian Bootstrap',
     xlab='Posterior Draws')
hist(PATE_ecdf_draws, xlim=c(-2,13), breaks=10, main='Empirical Dist. / MATE',
     xlab='Posterior Draws')
dev.off()

