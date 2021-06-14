# use_plsda.R
#
# Author: Xiuxia Du
# Date: Fall 2015, Fall 2016.






rm(list=ls())
graphics.off()





MLTB_dir <- "C:\\Users\\csa97\\Research\\Projects\\DuLab\\ADAP-ML\\adap-ml\\Xiuxia_old_scripts\\"








# -----------------------------------------------------------------------
# 1 PLS-DA illustrated using the iris data
# -----------------------------------------------------------------------

# 1.1 prepare the data
data(iris)

data.X <- as.matrix(iris[1:99, 1:4]) # n by d


data.y <- matrix(c(rep(-1, times=50), rep(1, times=49)), ncol=1) # n by 1
# +1 and -1 for class membership 





# 1.2 standardize X and y
# source(paste(MLTB_dir, "d_standardize.R", sep=.Platform$file.sep))
# X_standardized <- d_standardize(data.X)
# y_standardized <- d_standardize(data.y)
X_standardized <- scale(data.X)
y_standardized <- scale(data.y)




# 1.3 specify parameters for plsda
number_of_components <- 3
# number of PLS components





# 1.4 plsda
source(paste(MLTB_dir, "d_plsda.R", sep=.Platform$file.sep))
re.PLSDA <- d_plsda(X=X_standardized, y=y_standardized, c=number_of_components)


re.PLSDA$W
# unit w weight vectors
sqrt(sum(re.PLSDA$W[,1] * re.PLSDA$W[,1]))
sqrt(sum(re.PLSDA$W[,2] * re.PLSDA$W[,2]))
sqrt(sum(re.PLSDA$W[,3] * re.PLSDA$W[,3]))


# orthogonal w weight vectors
sum(re.PLSDA$W[,1] * re.PLSDA$W[,2])
sum(re.PLSDA$W[,1] * re.PLSDA$W[,3])
sum(re.PLSDA$W[,2] * re.PLSDA$W[,3])




re.PLSDA$T
sum(re.PLSDA$T[,1] * re.PLSDA$T[,2])
sum(re.PLSDA$T[,1] * re.PLSDA$T[,3])
sum(re.PLSDA$T[,2] * re.PLSDA$T[,3])
# orthogonal scores


re.PLSDA$P
re.PLSDA$Q
re.PLSDA$B







# 1.5 predict
y_hat <- X_standardized %*% re.PLSDA$B

plot(1:length(y_hat), y_hat,
     pch=16, cex=1,
     col="blue",
     xlab="sample index", ylab="prediction",
     main="prediction using a plsda model")
points(1:50, y_hat[1:50],
       pch=16, cex=1,
       col="red")
legend("topright", 
       legend=c("setosa", "versicolor"),
       pch=16, cex=1,
       col=c("red", "blue"))






# -----------------------------------------------------------------------
# 2. use the mixOmics r package to the iris data
# -----------------------------------------------------------------------

#packageName <- "mixOmics"
#install.packages(packageName)
#library(packageName, character.only = T)






# 1. apply the plsda function in the mixOmics package to the iris data

#re.PLSDA.mixOmics <- plsda(X=iris[1:99,1:4], Y=iris$Species[1:99], ncomp=3)
#str(re.PLSDA.mixOmics)

#head(re.PLSDA.mixOmics$X) 
# this is the standardized X
 

#head(re.PLSDA.mixOmics$Y) 
# this is the standardized y



#re.PLSDA.mixOmics$ind.mat
# indicator matrix, i.e. the y label matrix



#re.PLSDA.mixOmics$ncomp
# the number of components included in the model






# examine the X scores, i.e. the T matrix
#head(re.PLSDA.mixOmics$variates$X)
#head(re.PLSDA$T)
#re.PLSDA.mixOmics$variates$X / re.PLSDA$T
# The X scores are the same



# examine the Y scores, i.e. the U matrix
#head(re.PLSDA.mixOmics$variates$Y)




# examine the X loadings, i.e. the W matrix
#re.PLSDA.mixOmics$loadings$X 
#re.PLSDA$W
# the same X loadings


# examine the Y loadings
#re.PLSDA.mixOmics$loadings$Y 




#re.PLSDA.mixOmics$iter




# prediction using the plsda model obtained from the training phase
#re.predict.mixOmics <- predict(object=re.PLSDA.mixOmics, newdata=iris[100,1:4], method="centroids.dist")
#str(re.predict.mixOmics)
#re.predict.mixOmics$predict
#re.predict.mixOmics$variates
#re.predict.mixOmics$class















# -----------------------------------------------------------------------
# 3. use the DiscriMiner r package to the iris data
# -----------------------------------------------------------------------
#packageName <- "DiscriMiner"
#install.packages(packageName)
#library(packageName, character.only = T)

#rm(list=ls())



#DiscriMiner_R_dir <- "/Users/xdu4/Documents/Duxiuxia/Temp/test_DiscriMiner"

#source(paste(DiscriMiner_R_dir, "dd_plsDA.R", sep=.Platform$file.sep))
#source(paste(DiscriMiner_R_dir, "dd_my_verify.R", sep=.Platform$file.sep))
#source(paste(DiscriMiner_R_dir, "dd_my_plsDA.R", sep=.Platform$file.sep))
#source(paste(DiscriMiner_R_dir, "dd_my_tdc.R", sep=.Platform$file.sep))

#X <- iris[, 1:4]
#Y <- iris$Species



#re <- d_plsda_v2(X=X, Y=Y, cv="LOO")


#re.PLSDA.DiscriMiner <- dd_plsDA(X=X, Y=Y, autosel=T)

#str(re.PLSDA.DiscriMiner)

#re.PLSDA.DiscriMiner$functions  # discriminant function

#padded_iris <- as.matrix(cbind(rep(1, times=nrow(iris)), iris[,1:4]))
#projection <- padded_iris %*% re.PLSDA.DiscriMiner$functions    # this is the re.PLSDA.DiscriMiner$scores

#re.PLSDA.DiscriMiner$confusion

# the X score matrix, i.e. the T matrix
#re.PLSDA.DiscriMiner$scores


#re.PLSDA.DiscriMiner$loadings






#re.PLSDA.DiscriMiner$loadings

# examine the loadings
#sum(re.PLSDA.DiscriMiner$loadings[,1] * re.PLSDA.DiscriMiner$loadings[,2])
# loadings are not orthogonal


#re.PLSDA.DiscriMiner$y.loadings
#re.PLSDA.DiscriMiner$components



#plot(re.PLSDA.DiscriMiner)






#X <- iris[1:100,1:4]
#Y <- c(rep(1, 50), rep(-1, 50))

#re <- plsDA(variables=X, group=Y)



# -----------------------------------------------------------------------
# 4. use the pls r package to the iris data
# -----------------------------------------------------------------------
#packageName <- "pls"
#installl.packages(packageName)
#library(packageName, character.only = T)







# -----------------------------------------------------------------------
# 5. Misc
# -----------------------------------------------------------------------


#X <- iris[, 1:4]
#y <- iris$Species
#

#
#source(paste(MLTB_dir, "dd_my_tdc.R", sep=.Platform$file.sep))
#yy <- dd_my_tdc(y)
#
#

#re.dd_plsda <- dd_plsDA(X=iris[1:N,1:4], 
#                        y=iris$Species[1:N], 
#                        learn=1:N,
#                        test=1:N,
#                       autosel=T,
#                        cv = "LOO", 
#                        retain.models = FALSE)





# -----------------------------------------------------------------------
# 6. V2 of d_plsda
# -----------------------------------------------------------------------
#

#X_standardized <- scale(X)
#
#source
#yy <- 
#
#source(paste(MLTB_dir, "d_stand.R", sep=.Platform$file.sep))

