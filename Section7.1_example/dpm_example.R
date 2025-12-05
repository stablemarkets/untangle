library(rstan)
library(scales)

n <- 300 
set.seed(3)

l = seq(1,pi, length.out = n) + rnorm(n,0,.1) # confounder
a = rbinom(n, 1, plogis( -2 + .1*l) )
y0 = rnorm( n = length(l), sin(3*l), .1*l)
y1 = rnorm( n = length(l), sin(2*l) + .5*(y0 - sin(3*l)), .07*l)

y = a*y1 + (1-a)*y0

d = data.frame(y = y,
               a = a,
               l = l)

# plot(l, y0)
# points(l, y1, col='red')
# plot(l, y, col=ifelse(a==0, 'black','red'))

## load stan model
model = stan_model("dpm_example.stan")

n_1 = sum(d$a)
n_0 = n - n_1

grid = seq(min(d$l), max(d$l), length.out=1000)

## data to pass to Stan
stan_data = list(n=n, n_1 = n_1, n_0=n_0,
                 y_1 = d$y[d$a==1], 
                 l_1 = d$l[d$a==1], 
                 y_0 = d$y[d$a==0],
                 l_0 = d$l[d$a==0],
                 ## for plotting regression line
                 l_test = grid,
                 N_test = length(grid),
                 K = 10)

## run Stan model
fit = sampling(model, seed=3,data = stan_data, iter = 2000, warmup=1000, chains = 1)

## extract posterior draws of E[Y|X] for each X in test set.
post1 = extract(fit, pars= c('cond_mean_y_a1') )$cond_mean_y_a1
post0 = extract(fit, pars= c('cond_mean_y_a0') )$cond_mean_y_a0


png("dpm_example.png", width=800, height=800)
par(mfrow=c(2,2))

plot(d$l[d$a==0], d$y[d$a==0], xlim=c(.9,3.3), pch=20,col=alpha('black',alpha=.2),
     xlab='L', ylab='Y', ylim=c(-1.5,1.5), 
     main='Posterior Draws of E[Y|A=a,L=l] \nDPM Model')

legend('bottomleft', 
       legend=c('a=1','a=0'),
       col = c('steelblue','black'), bty='n', lty=1, lwd=2 )

for(j in 1:50){
  lines(grid, post0[100+j,], col=alpha('darkgray',alpha=.8), pch=20)
}
lines(grid, apply(post0,2,mean), col='black', lwd=2) 

for(j in 1:50){
  lines(grid, post1[100+j,], col=alpha('steelblue',alpha=.5), pch=20)
}
lines(grid, apply(post1,2,mean), col='darkblue', lwd=2) 

points(d$l[d$a==1], d$y[d$a==1], col=alpha('lightblue',alpha=.5),, pch=20)
points(d$l[d$a==0], d$y[d$a==0], col=alpha('black',alpha=.2),, pch=20)


cate = post1 - post0
post_mean_cate = apply(cate,2,mean)
plot(grid, post_mean_cate, xlim=c(.9,3.3), type='l',col='black',
     xlab='L', ylab='CATE', ylim=c(-2.5,2), 
     main='Posterior Draws of CATE \nDPM Model')

for(j in 1:50){
  lines(grid, cate[100+j,], col=alpha('steelblue',alpha=.5), pch=20)
}
lines(grid, post_mean_cate, col='darkblue', lwd=2)
lines(grid, sin(2*grid)-sin(3*grid), col='red', lwd=2)

legend('bottomleft', 
       legend=c('CATE Posterior','True CATE'),
       col = c('steelblue','red'), bty='n', lty=1, lwd=2 )

y1_m = extract(fit, pars= c('y_1m') )$y_1m
y0_m = extract(fit, pars= c('y_0m') )$y_0m

ite_1 = apply(y0_m, 1, function(x) d$y[d$a==1] - x )
ite_0 = apply(y1_m, 1, function(x) x - d$y[d$a==0])
ite = matrix(NA, nrow=n, ncol=1000)
ite[d$a==1, ] = ite_1
ite[d$a==0, ] = ite_0

plot(d$l, apply(ite, 1, mean), ylim=c(-3,2), col='steelblue', pch=20, 
     xlim = c(.9, 3.1), 
     xlab = 'L_i, i=1,2,...,n', ylab='ITE of Each Unit, i',
     main = 'Posterior of ITEs \nDPM Model')
lwr = apply(ite, 1, quantile, prob=.05)
upr = apply(ite, 1, quantile, prob=.95)
segments(d$l, lwr, d$l, upr, col='steelblue')

ite_true = y1 - y0
points(d$l,ite_true, pch=20, col='red',cex=.5)

legend('bottomleft', 
       legend=c('ITE Posterior','True ITE'),
       col = c('steelblue','red'), bty='n', 
       lty=c(1,NA), lwd=c(2,NA), pch=c(20, 20)  )

postl = extract(fit, pars= c('cond_density_l') )$cond_density_l

hist(d$l, freq=F, ylim=c(0,1), 
     main='Posterior of Covariate Distribution \nDPM Model', 
     xlab='L')
for(j in 1:50){
  lines(grid, postl[100+j,], col=alpha('steelblue',alpha=.5), pch=20)
}
lines(grid, apply(postl,2, mean), col='blue', lwd=2)

legend('topleft', 
       legend=c('Posterior of Covariate Density','Observed Data'),
       col = c('steelblue','gray'), bty='n', 
       lty=c(1,NA), lwd=c(2,NA), pch=c(NA, 15)  )

dev.off()

