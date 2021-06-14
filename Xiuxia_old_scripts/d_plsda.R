d_plsda <- function(X, y, c) {
    
    # X is the input data in an n by d matrix. n is the number of samples
    # y is the dependent variable in an n by 1 column vector
    
    
    
    # initialize data holders
    X_iter <- vector(mode="list", length=c) # n by d
    y_iter <- vector(mode="list", length=c)
    
    w_iter <- vector(mode="list", length=c) # d by 1
    t_iter <- vector(mode="list", length=c) # n by 1
    u_iter <- vector(mode="list", length=c)
    p_iter <- vector(mode="list", length=c) # d by 1
    
    
    ts_iter <- vector(mode="list", length=c) # scalar
    qs_iter <- vector(mode="list", length=c) # scalar
    
    
    X_iter[[1]] <- X # n by d
    y_iter[[1]] <- y
    
    w_iter[[1]] <- t(X) %*% y # an initial estimate of the weight vector
    w_iter[[1]] <- w_iter[[1]] / sqrt(sum(w_iter[[1]]^2)) # normalize weight vector
    
    t_iter[[1]] <- X %*% w_iter[[1]] # X scores
    
    
    
    
    # iteration
    for (k in 1:c) {
        ts_iter[[k]] <- sum( t_iter[[k]] * t_iter[[k]] ) # scalar
        
        t_iter[[k]] <- t_iter[[k]] / ts_iter[[k]]
        
        p_iter[[k]] <- t(X_iter[[k]]) %*% t_iter[[k]] # X loadings
        
        qs_iter[[k]] <- sum( y * t_iter[[k]] ) # y loadings
        
        if (qs_iter[[k]] == 0) {
            c <- k
            break
        }
        
        if (k < c) {
            X_iter[[k+1]] <- X_iter[[k]] - ts_iter[[k]] * t_iter[[k]] %*% t(p_iter[[k]])
            # subtract the effect of the new pLS component from the data to obtain a residual data matrix
            
            y_iter[[k+1]] <- y_iter[[k]] - ts_iter[[k]] * t_iter[[k]] * qs_iter[[k]]
            # residual y
            
            w_iter[[k+1]] <- t(X_iter[[k+1]]) %*% y_iter[[k+1]]
            # update the weight vector
            
            w_iter[[k+1]] <- w_iter[[k+1]] / sqrt(sum(w_iter[[k+1]] * w_iter[[k+1]]))
            # normalize the weight vector
            
            t_iter[[k+1]] <- X_iter[[k+1]] %*% w_iter[[k+1]]
            # update the X score
        }
    }
    
    
    W <- matrix(0, nrow=ncol(X), ncol=c)    # transformation matrix for transforming X
    T <- matrix(0, nrow=nrow(X), ncol=c)    # projection of X, i.e. the latent score
    P <- matrix(0, nrow=ncol(X), ncol=c)    # X loading matrix
    Q <- vector(mode="numeric", length=c)   # Y loading matrix
    
    for (i in 1:c) {
        W[,i] <- w_iter[[i]]
        T[,i] <- t_iter[[i]]
        P[,i] <- p_iter[[i]]
        Q[i] <- qs_iter[[i]]
    }
    
    
    B <- (W %*% solve(t(P) %*% W)) %*% Q    # for prediction
    
    re <- list()
    
    re$W <- W
    re$T <- T   
    re$P <- P
    re$Q <- Q
    
    re$B <- B
    
    
    return(re)
} 








d_loocv_plsda <- function(x, y, c) {
    
    re <- vector(mode="list", length=nrow(x))
    
    
    for (i in 1:nrow(x)) {
        
        if (i==1) {
            x.test <- x[1,]
            x.test.labels <- y[1]
            
            x.training <- x[2:nrow(x),]
            x.training.labels <- y[2:nrow(x)]
            
        } else if (i==nrow(x)) {
            x.test <- x[i,]
            x.test.labels <- y[i]
            
            x.training <- x[1:(nrow(x)-1),]
            x.training.labels <- y[1:(nrow(x)-1)]
            
        } else {
            x.test <- x[i,]
            x.test.labels <- y[i]
            
            x.training <- rbind( x[1:(i-1),], x[(i+1):nrow(x),] )
            x.training.labels <- c(y[1:(i-1)], y[(i+1):nrow(x)])
        }
        
        
        
        
        
        re.plsda <- d_plsda(x, y, c)
        
        
        
        predict <- x.test %*% re.plsda$B
        
        
        if (predict * x.test.labels > 0) {
            re[[i]]$error_rate <- 0
        } else {
            re[[i]]$error_rate <- 1
        }
        
    }
    
    
    
    error_rate_vector <- vector(mode="numeric", length=nrow(x))
    
    for (i in 1:nrow(x)) {
        error_rate_vector[i] <- re[[i]]$error_rate
    }
    
    overall_error_rate <- mean(error_rate_vector)
    
    return(list(result_list=re, error_rate_vector=error_rate_vector, overall_error_rate=overall_error_rate))
}