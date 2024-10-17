set.seed(759)


# Function to calculate the logarithm of the Gaussian pdf at `x`
logpi_gauss <- function(x, mu = 0, sigma = 1) {
  dnorm(x, mean = mu, sd = sigma, log = TRUE)
}


# Function to implement the Random Walk Metropolis algorithm for a d-dimensional
# target distribution
RWM <- function(logpi, n_iter, h, x_curr) {
  # Number of dimensions
  d <- length(x_curr)

  # Calculate the logarithm of the target distribution at the current state
  logpi_curr <- logpi(x_curr)

  # Counter for accepted proposals
  accepted <- 0

  # Matrix to store the sampled values from the chain
  x_store <- matrix(NA, nrow = d, ncol = n_iter)

  for (i in 1:n_iter) {
    # Generate a candidate move
    x_prop <- x_curr + h * rnorm(1)
    logpi_prop <- logpi(x_prop)

    # Calculate the difference in the log-probabilities of the proposed and
    # current states (i.e. \log \frac{\pi(y)}{\pi(x)})
    loga <- logpi_prop - logpi_curr

    # Generate a random uniform number between 0 and 1
    u <- runif(1)

    # Acceptance criterion (log-space comparison for numerical stability)
    if (all(log(u) < loga)) {
      # If accepted, update the current state and log-probability
      x_curr <- x_prop
      logpi_curr <- logpi_prop
      accepted <- accepted + 1
    }
    x_store[, i] <- x_curr
  }

  return(list(x_store = x_store, a_rate = accepted / n_iter))
}


n_iter <- 2000
step_size <- 2
current_state <- rep(0, 2)

# Run the function to generate a Markov chain
mc <- RWM(
  logpi = logpi_gauss,
  n_iter = n_iter,
  h = step_size,
  x_curr = current_state
)


# Plot the chain and output the acceptance rate
plot(1:n_iter, mc$x_store, type = "l", xlab = "t", ylab = "X(t)")
cat("The acceptance rate is: ", mc$a_rate)
