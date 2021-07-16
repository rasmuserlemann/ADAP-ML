d_plsda_v2 <- function(X, Y, c) {
    
    # X is the input data in an n by d matrix. n is the number of samples
    # y is the dependent variable in an n by 1 column vector
    # iter is the number of iterations
    
    
    
    
    
    Y <- dd_my_tdc(data.frame(Y)) 
    
    
    # get dimensions
    n <- nrow(X)    # number of samples
    d <- ncol(X)    # dimension of X
    q <- ncol(Y)    # dimension of y
    
    
    
    # determine number of PLS components to be computed
    X_svd <- svd(X, nu=0, nv=0)
    rank_X <- sum(X_svd$d > 0.0001)
    
    if (rank_X ==0) {
        stop("\nrank = 0: varaibles are numerically constant")
    }
    
    c <- min(n, rank_X)    # number of PLS components
    
    if (c == n) {
        c <- n-1
    }
    
    
    
    
    # standardizing data
    X_old <- scale(X)
    Y_old <- scale(Y)
    
    
    
    
    # initialize data holders
    W <- matrix(0, nrow=d, ncol=c)      # 
    T <- matrix(0, nrow=n, ncol=c)      # X scores
    P <- matrix(0, nrow=d, ncol=c)      # X loadings
    U <- matrix(0, nrow=n, ncol=c)      # Y scores
    Q <- matrix(0, nrow=p, ncol=c)      # Y loadings
    B <- rep(0, c)
    
    
    RSS <- rbind(rep(n-1, p), matrix(NA, c, p))
    PRESS <- matrix(NA, c, p)
    Q2 <- matrix(NA, c, p)
    
    
    
    # X_iter <- vector(mode="list", length=iter) # n by d
    # Y_iter <- vector(mode="list", length=iter)
    # 
    # W_iter <- vector(mode="list", length=iter) # d by 1
    # T_iter <- vector(mode="list", length=iter) # n by 1
    # U_iter <- vector(mode="list", length=iter)
    # P_iter <- vector(mode="list", length=iter) # d by 1
    # 
    # 
    # ts_iter <- vector(mode="list", length=iter) # scalar
    # qs_iter <- vector(mode="list", length=iter) # scalar
    
    
    
    
    # X_iter[[1]] <- X # n by d
    # Y_iter[[1]] <- Y
    # 
    # w_iter[[1]] <- t(X) %*% Y # an initial estimate of the weight vector
    # w_iter[[1]] <- w_iter[[1]] / sqrt(sum(w_iter[[1]]^2)) # normalize weight vector
    # 
    # t_iter[[1]] <- X %*% w_iter[[1]] # X scores
    
    
    
    
    # PLS2 algorithm
    for (h in 1:c) {
        u_new <- Y_old[,1]
        w_old <- rep(1, d)
        
        iter <- 1
        
        repeat{
            w_new <- t(X_old) %*% u_new / sum(u.new^2)
            w_new <- w_new / sqrt(sum(w_new^2)) # normalize w_new
            
            t_new <- X_old %*% w_new            # project X onto w
            
            q_new <- t(Y_old) %*% t_new / sum(t_new^2)
            u_new <- Y_old %*% q_new / sum(q_new^2)
            
            w_diff <- w_new - w_old
            w_old <- w_new
            
            if (sum(w_diff^2) < 1e-06 || iter==100) {
                break
            }
            
            iter <- iter + 1
        }
        p_new <- t(X_old) %*% t_new / sum(t_new^2)
        
        
        # cross validation
        RSS[h+1, ] <- colSums((Y_old - t_new %*% t(q_new))^2)
        press <- matrix(0, n, p)
        
        
        if (cv == "LOO") {
            for (i in 1:n) {
                uh_si <- Y_old[-i, 1]
                wh_si_old <- rep(1, d)
                
                iter_cv <- 1
                
                repeat {
                    wh_si <- t(X_old[-i,]) %*% uh_si / sum(uh_si^2)
                    wh_si <- wh_si / sqrt(sum(wh_si^2))
                    
                    th_si <- X_old[-i,] %*% wh_si
                    
                    qh_si <- t(Y_old[-i,]) %*% th_si / sum(th_si^2)
                    uh_si <- Y_old[-i,] %*% qh_si / sum(qh_si^2)
                    
                    w_si_diff <- wh_si - wh_si_old
                    
                    wh_si_old <- wh_si
                    
                    if(sum(w_si_diff^2) < 1e-06 || iter_cv == 100) {
                        break
                    }
                    
                    iter_cv <- iter_cv + 1
                }
            }
        }
    }
    
    
    
    

    
    
    
    
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