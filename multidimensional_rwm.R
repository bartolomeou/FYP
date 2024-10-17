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

  # Matrix to store the sampled values from the chain (#components, #iterations)
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


# Run the function to generate a Markov chain
mc <- RWM(
  logpi = logpi_gauss,
  n_iter = 2000,
  h = 2,
  x_curr = rep(0, 2)
)

# Plot the chain and output the acceptance rate
plot(1:n_iter, mc$x_store[1, ], type = "l", xlab = "t", ylab = "X(t)")
cat("The acceptance rate is: ", mc$a_rate)


# Testing different step sizes and examining its effect on the acceptance rate
start <- 0.001 # Starting value for the smallest step size
multiplier <- 2 # Multiplier to increase the step size
n <- 8 # Number of step sizes
step_size_grid <- start * multiplier^(0:(n - 1))

n_dim <- 2 # Dimension of the target distribution
n_iter <- 200

# Array to store samples (#step sizes, #dimensions, #iterations)
mc_samples <- array(NA, dim = c(n, n_dim, n_iter))
# Vector to store acceptance rates for each step size
mc_a_rates <- rep(NA, n)

# Run the RMW algorithm for each step size
for (i in 1:length(step_size_grid)) {
  mc <- RWM(
    logpi = logpi_gauss,
    n_iter = n_iter,
    h = step_size_grid[i],
    x_curr = rep(0, n_dim)
  )

  mc_samples[i, , ] <- mc$x_store
  mc_a_rates[i] <- mc$a_rate
}

# Plot the acceptance rates against step sizes on a log scale
plot(step_size_grid, mc_a_rates, log = "x", xaxt = 'n', 
     xlab = "Step size", ylab = "Acceptance rate", pch=20)
axis(1, at = step_size_grid)

matplot(1:n_iter, t(mc_samples[, 1, ]), type = "l", 
        xlab = "t", ylab = expression(X[1](t)), 
        col = 1:n, lty = 1, ylim = range(mc_samples[, 1, ]))
legend("bottom", legend = step_size_grid, horiz = TRUE, 
       col = 1:n, lty = 1, lwd = 2, cex = 0.7, bty = "n")
