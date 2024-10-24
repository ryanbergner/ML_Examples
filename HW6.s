# Log likelihood
log_likelihood <- function(beta, X, y) {
  n <- length(y)
  sigma_sq <- 1 # Fixed value
  y_hat <- X %*% beta
  res <- y - y_hat
  return(-0.5 * n * log(2 * pi * sigma_sq) - 0.5 * (1 / sigma_sq) * t(res) %*% res)
}

# Log prior
log_prior <- function(beta) {
  return(sum(dnorm(beta, mean=0, sd=1, log=TRUE)))
}

# Log posterior (up to a normalizing constant)
log_posterior <- function(beta, X, y) {
  return(log_likelihood(beta, X, y) + log_prior(beta))
}

# Prepare the data
X <- cbind(1, mtcars$weight, mtcars$horsepower)
y <- mtcars$mpg
data <- list(X=X, y=y)