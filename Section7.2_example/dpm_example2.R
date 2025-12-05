library(rstan)
library(scales)
library(latex2exp)

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
model = stan_model("dpm_example2.stan")

n_1 = sum(d$a)
n_0 = n - n_1

grid = seq(min(d$l), max(d$l), length.out=30)

## data to pass to Stan
stan_data = list(n=n, n_1 = n_1, n_0=n_0,
                 y_1 = d$y[d$a==1], 
                 l_1 = d$l[d$a==1], 
                 y_0 = d$y[d$a==0],
                 l_0 = d$l[d$a==0],
                 ## for plotting regression line
                 l_test = grid,
                 N_test = length(grid),
                 K = 10, 
                 rho=0, 
                 B=500)

## run Stan model
fit = sampling(model, seed=3,data = stan_data, iter = 2000, warmup=1000, chains = 1)

## extract posterior draws of E[Y|X] for each X in test set.
post = extract(fit, pars= c('cond_rate') )$cond_rate

posty1 = extract(fit, pars= c('y1_sim') )$y1_sim
posty0 = extract(fit, pars= c('y0_sim') )$y0_sim

png('dpm_example2.png', width=800, height=400)
par(mfrow=c(1,2))
plot(grid, apply(post,2, mean), type='p', pch=20, 
     xlab='Grid of L=l values', ylab=TeX('$\\psi_1(l)$'), 
     main = TeX('Posterior Draws of $\\psi_1(l)$') )
for(j in 1:50){
  points(grid, post[50+j,], col=alpha('lightblue',alpha=.2), pch=20)
}

points(grid, apply(post,2, mean), pch=20, cex=1.5, col='steelblue')

plot_unit = 6
plot_draw = 706

points(grid[plot_unit], post[plot_draw, plot_unit], col=alpha('red',alpha=.9), pch=20 )
val = post[plot_draw, plot_unit]
text( grid[plot_unit]+.05, val+.05, val, col='red')


mc_drawsy0 = posty0[plot_draw, plot_unit,]
mc_drawsy1 = posty1[plot_draw, plot_unit,]

plot(mc_drawsy0, mc_drawsy1, xlim=c(-3, 3), ylim=c(-3,3),, 
     xlab='Y(0)',ylab='Y(1)', 
     main='MC Draws of (Y(0),Y(1)) at L=1.3, t=706')

rect(xleft   = -.5,
     ybottom = -.5,
     xright  = 6,
     ytop    = 6,
     col = "gray90", border = NA)
points(mc_drawsy0, mc_drawsy1)
points(mc_drawsy0, mc_drawsy1, col= ifelse(mc_drawsy1>-.5 & mc_drawsy0 >-.5, 'red', 'NA'))

abline(h=0, lty=2)
abline(v=0, lty=2)

text(.1,1.5, paste0(' of draws land in shaded area'))
text(-2,1.5, paste0(val*100,'% '), col='red')
dev.off()

mean(mc_drawsy1>-.5 & mc_drawsy0 >-.5)

