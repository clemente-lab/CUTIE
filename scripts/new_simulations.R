###
# Libraries
###

library(readr)
library(ggplot2)
library('reshape2')
library("optparse")
#library(e1071)
library('MASS')

option_list = list(
  make_option(c("--n_sampvec"), default=NULL,type="character",
              help="vector of n_samps to use"),
  make_option(c("--max_seed"), default=NULL,type="integer",
              help="max seed value (inclusive)"),
  make_option(c("--start"), default=NULL,type="double",
              help="start value for corrs"),
  make_option(c("--stop"), default=NULL,type="double",
              help="stop value for corrs (inclusive)"),
  make_option(c("--step"), default=NULL,type="double",
              help="step value for corrs"),
  make_option(c("--output"), type="character", default=NULL,
              help="outputdir", metavar="character")
  
); 
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser)

print(opt)

attach(opt)
for (n_samp in strsplit(n_sampvec,split=',')[[1]]){
  n_samp = as.integer(n_samp)
  for (nseed in seq(from=0, to=max_seed, by=1)){
    # FP/FN/P
    # 'nseed_class_corr_nsamp'
    
    for (cv in seq(from = start, to = stop, by = step)) { 
      set.seed(nseed)
      data = mvrnorm(n=(n_samp), mu=c(0, 0), Sigma=matrix(c(1, cv, cv, 1), nrow=2), empirical=TRUE)
      X = data[, 1]  # standard normal (mu=0, sd=1)
      Y = data[, 2]  # standard normal (mu=0, sd=1)
      S = seq(1:n_samp)
      mat = cbind(S,Y,X)
      #png(filename=paste("Desktop/clemente_lab/CUTIE/plots/P_", cv,'_', cor(X,Y),"_plot.png", sep=''))
      #pairs(cbind(X,Y))
      write.table(mat, file=paste(output, nseed,'_NP_',n_samp,'_',cv,'.txt',sep=''), row.names=sprintf("s%s",seq(1:n_samp)), col.names=TRUE, sep='\t')
      #dev.off()
      #mat = cbind(S,log(X),log(Y))
      #png(filename=paste("Desktop/clemente_lab/CUTIE/plots/P_", cv,'_', cor(X,Y),"_plot.png", sep=''))
      #pairs(cbind(X,Y))
      #write.table(mat, file=paste(output, nseed,'_NP_',n_samp,'_',cv,'.txt',sep=''), row.names=sprintf("s%s",seq(1:n_samp)), col.names=TRUE, sep='\t')
      
    }
  
    
    # AQ FN case
    for (cv in seq(from = start, to = stop, by = step)) { 
      set.seed(nseed)
      data = mvrnorm(n=n_samp-1, mu=c(0, 0), Sigma=matrix(c(1, cv, cv, 1), nrow=2), empirical=TRUE)
      X = data[, 1]  # standard normal (mu=0, sd=1)
      Y = data[, 2]  # standard normal (mu=0, sd=1)
      X <- c(X, 3)
      Y <- c(Y, -3)
      S = seq(1:n_samp)
      mat = cbind(S,Y,X)
      write.table(mat, file=paste(output, nseed,'_FN_',n_samp,'_',cv,'.txt',sep=''), row.names=sprintf("s%s",seq(1:n_samp)), col.names=TRUE, sep='\t')
      #png(filename=paste("Desktop/clemente_lab/CUTIE/plots/FN_", cv,'_', cor(X,Y),"_plot.png", sep=''))
      #pairs(cbind(X,Y))
      #dev.off()
    }
    
    
    # try with anscombe's q
    # https://stat.ethz.ch/pipermail/r-help/2007-April/128925.html
    for (cv in seq(from = start, to = stop, by = step)) { 
      set.seed(nseed)
      data = mvrnorm(n=n_samp-1, mu=c(0, 0), Sigma=matrix(c(1, 0, 0, 1), nrow=2), empirical=TRUE)
      X = data[, 1]  # standard normal (mu=0, sd=1)
      Y = data[, 2]  # standard normal (mu=0, sd=1)
      # empirically determine correlation within error
      eps = 0.01
      for (i in exp(seq(from=-4,to=10,by=0.01))){
        x1 <- c(X, 20)
        x2 <- c(Y, i)
        corr <- cor(x1,x2)
        if (abs(corr-cv) < eps) {
          break
        }
      }
      X <- x1
      Y <- x2
      if (cv == 1){
        X <- numeric(n_samp-1)
        Y <- numeric(n_samp-1)
        X <- c(X, 20)
        Y <- c(Y, 20)
      }
      S = seq(1:n_samp)
      mat = cbind(S,Y,X)
      write.table(mat, file=paste(output, nseed,'_FP_',n_samp,'_',cv,'.txt',sep=''), row.names=sprintf("s%s",seq(1:n_samp)), col.names=TRUE, sep='\t')
    }
    
    # CD examples
    for (cv in seq(from = start, to = stop, by = step)) { 
      set.seed(nseed)
      data = mvrnorm(n=(n_samp-1), mu=c(0, 0), Sigma=matrix(c(1, cv, cv, 1), nrow=2), empirical=TRUE)
      X = data[, 1]  # standard normal (mu=0, sd=1)
      Y = data[, 2]  # standard normal (mu=0, sd=1)
      theta = atan(cv) + atan(1/2)
      X <- c(X,7*cos(theta))
      Y <- c(Y,7*sin(theta))
      S = seq(1:n_samp)
      mat = cbind(S,Y,X)
      # pairs(cbind(X,Y))
      write.table(mat, file=paste(output, nseed,'_CD_',n_samp,'_',cv,'.txt',sep=''), row.names=sprintf("s%s",seq(1:n_samp)), col.names=TRUE, sep='\t')
      # dev.off()
      #model = lm(Y~X)
      #print(paste(cv, cooks.distance(model)[n_samp], unname(cor.test(X,Y, method='pearson')$estimate), cor.test(X,Y, method='pearson')$p.value))
    }
    
    
  }
}
