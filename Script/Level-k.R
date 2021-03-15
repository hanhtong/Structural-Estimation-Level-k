# R code for structural estimation of level-k model for a research on individual heterogeneity in reasoning.
# The study used Traveler's Dilemma game.
# Author: Hanh Tong and David Freeman

# The study considered 15 models for strategic reasoning.
# The code below is for one of the 15 models.
# Model: Level-k, with L0 playing the upper bound of the guessing range, and uniform errors.
# The errors follow uniform distribution, to resemble "trembling hand" errors.

# For the structural estimation of this model, we need:
# (1) a function to calculate best responses with uniform errors incorporated.
# (2) a function to calculate the level distribution
# (3) a function that combines actual choices, functions in (1) and (2) to form log-likelihood function
# (4) an optimization routine using maxLik to maximize the log-likelihood function from (3).


#--------------------------------------------------------------------------
# Get the packages 
install.packages('pacman') 
pacman::p_load(doParallel, doSNOW, foreach, maxLik)

parallel:::detectCores()
clust <- makeCluster(parallel:::detectCores()-1)
registerDoSNOW (clust)

# Set path and working directory 
# path = "Structural-Estimation-Level-k" # replace with the path you choose
setwd(path)

# Read in the data
dat = read.csv("Data/Data_TD.csv")
dat = dat[, -1] # drop the first column, subjects' responses are from columns 2 to 61
# each column records 30 responses for 30 rounds of the game, for each subject

# Game parameter 
Game_para = matrix(nrow = 30, ncol = 3)
colnames(Game_para)=c("lower bound", "upper bound", "reward/penalty")
# reward/penalty para
Game_para[ , 3]= c(5,5,5,5,5,5,5,5,5,5,
                   20,20,20,20,20,20,20,20,20,20,
                   5,5,5,5,5,5,5,5,5,5)
# lower bound
Game_para[ , 1]= c(20,80,40,20,50,20,60,100,50,40,
                   20,80,40,20,50,20,60,100,50,40,
                   20,80,40,20,50,20,60,100,50,40)
# upper bound
Game_para[ , 2]= c(120,200,160,180,200,160,180,200,150,200,
                   120,200,160,180,200,160,180,200,150,200,
                   120,200,160,180,200,160,180,200,150,200)

#---------------------------------------------------------------------------------
# Lk_br is a function that calculates the theoretical best response for level-k, with L0 playing the upper bound
# then add uniform error to form noisy best response
Lk_br = function(eps,j){  #for game j
  game_range = Game_para[j,2] - Game_para[j,1] + 1
  Level_br = matrix(0,game_range,game_range) 
  for (m in 1:game_range){
    Level_br[game_range-m+1,m] = 1
  }
  unoiseLK =(1-eps)*Level_br + eps/game_range #add uniform errors to best responses
  return(unoiseLK)
}
#------------------------------------------------------------------------------------------------------------------
# Function mix_level calculate the probability distribution of level.
# The level distribution is modelled to follow Poisson distribution.
# This approach to modelling level distribution allows us to avoid estimating the model multiple times
# tau is the level distribution parameter
mix_level = function(tau, j){ #for game j
  game_range = Game_para[j,2] - Game_para[j,1] + 1
  ex= matrix(0, game_range,1)
  mix= matrix(0, game_range,1)
  for (i in 1:game_range){
    ex[i,1] = ((exp(-tau))*(tau^(i-1)))/factorial(i-1)
  }
  for (h in 1:game_range){
    mix[h,1] = ex[h,1]/sum(ex)
  }
  return(mix)
} 
#------------------------------------------------------------------------------------------------------------------
# Function P_LK takes actual choices and return the probability of observing the choices for each level
P_LK = function(eps, tau,i){
  P_LK = matrix(NA, 30,1)
  for (j in 1:30) {
    best = Lk_br(eps,j)
    mix= mix_level(tau,j) 
    choice = dat[j, i] - Game_para[j,1] + 1
    prob = best[choice, ]%*%mix
    P_LK[j,1]= log(prob)
  }
  return(P_LK)
}
#------------------------------------------------------------------------------------------------------------------
# This part of the code performs maximum likelihood estimations for each subject,
# and return a csv file recording the estimation results
n_subject = ncol(dat)
LK_uniform_sub = matrix(NA, n_subject, 8)
colnames(LK_uniform_sub)=c("eps","s.e eps","p-value eps","tau", "s.e tau","p-value tau", "LogLik","AIC")

foreach (i = 1:n_subject)%do% {
  LL = function(theta){
    eps = theta[1]
    tau = theta[2]
    prob = P_LK(eps, tau, i )
    l = sum(prob)
    return(l)
  }
  
  #constraints on estimated parameters
  A = matrix(c(1,-1,0,0,0,1), 3,2)
  B = matrix(c(0,1,0),3,1)
  mle_Level = maxLik(logLik = LL, start = c(eps = 0.1, tau = 1), method = "BFGS",constraints = list(ineqA=A, ineqB =B))
  
  LK_uniform_sub[i,1] = summary(mle_Level)$estimate[1] #eps
  LK_uniform_sub[i,4] = summary(mle_Level)$estimate[2] #tau
  LK_uniform_sub[i,2] = summary(mle_Level)$estimate[3] #std. error for eps
  LK_uniform_sub[i,5] = summary(mle_Level)$estimate[4] #std. error for tau
  LK_uniform_sub[i,3] = summary(mle_Level)$estimate[7] #p-value for eps
  LK_uniform_sub[i,6] = summary(mle_Level)$estimate[8] #p-value for tau
  LK_uniform_sub[i,7] = logLik(mle_Level) # loglik
  LK_uniform_sub[i,8] = AIC(mle_Level) #AIC
}

LK_uniform_sub = round(LK_uniform_sub,digits = 5)
LK_uniform_sub = as.data.frame(LK_uniform_sub)
write.csv(LK_uniform_sub,file = "LK_top_uniform error.csv",row.names = T,col.names = T)





