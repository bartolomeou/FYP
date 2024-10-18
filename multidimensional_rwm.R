library(RColorBrewer)
set.seed(759)



### BUILDING A MULTI-DIMENSIONAL RWM ALGORITHM
# Function to calculate the logarithm of the Gaussian pdf at `x`
logpi_gauss <- function(x, location = 0, scale = 1) {
  sum(dnorm(x, mean = location, sd = scale, log = TRUE))
}


# Function to implement the Random Walk Metropolis algorithm for a d-dimensional
# target distribution
RWM <- function(logpi, n_iter, h, x_curr, ...) {
  # Number of dimensions
  d <- length(x_curr)

  # Calculate the logarithm of the target distribution at the current state
  logpi_curr <- logpi(x_curr, ...)

  # Counter for accepted proposals
  accepted <- 0

  # Matrix to store the sampled values from the chain (#components, #iterations)
  x_store <- matrix(NA, nrow = d, ncol = n_iter)

  for (i in 1:n_iter) {
    # Propose a candidate move
    x_prop <- x_curr + h * rnorm(d)
    logpi_prop <- logpi(x_prop, ...)

    # Calculate the difference in the log-probabilities of the proposed and
    # current states (i.e. \log \frac{\pi(y)}{\pi(x)})
    loga <- logpi_prop - logpi_curr

    # Generate a random uniform number between 0 and 1
    u <- runif(1)

    # Acceptance criterion (log-space comparison for numerical stability)
    if (log(u) < loga) {
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
plot(1:2000, mc$x_store[1, ], type = "l", xlab = "t", ylab = "X(t)")
cat("The acceptance rate is: ", mc$a_rate)


# Testing different step sizes and examining its effect on the acceptance rate
start <- 0.01 # Starting value for the smallest step size
multiplier <- 2.5 # Multiplier to increase the step size
n <- 8 # Number of step sizes
step_size_grid <- signif(start * multiplier^(0:(n - 1)), digits = 2)

n_dim <- 2 # Dimension of the target distribution
n_iter <- 2000

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
plot(step_size_grid, mc_a_rates,
  log = "x", xaxt = "n",
  xlab = "Step size", ylab = "Acceptance rate", pch = 20
)
axis(1, at = step_size_grid)

matplot(1:n_iter, t(mc_samples[, 1, ]),
  ylim = range(mc_samples), type = "l",
  xlab = "t", ylab = expression(X[1](t)),
  col = 1:n, lty = 1
)
legend("bottom",
  legend = step_size_grid, ncol = ceiling(n / 2),
  col = 1:n, lty = 1, lwd = 2, cex = 0.7, bty = "n"
)


# 10-dimensional Gaussian target distribution
n_iter <- 2000
step_size <- 0.8
mc_10d <- RWM(logpi = logpi_gauss, n_iter = n_iter, h = step_size, x_curr = rep(0, 10))
cat("The acceptance rate is", mc_10d$a_rate)



### UNDERSTANDING TAIL BEHAVIOUR
library(VGAM)


# Function to calculate the logarithm of the Laplace pdf at `x`
logpi_laplace <- function(x, location = 0, scale = 1) {
  sum(dlaplace(x, location = location, scale = scale, log = TRUE))
}


# Function to calculate the logarithm of the Cauchy pdf at `x`
logpi_cauchy <- function(x, location = 0, scale = 1) {
  sum(dcauchy(x, location = location, scale = scale, log = TRUE))
}


# Warm start
initial <- 0
n_iter <- 500

mc_g <- RWM(logpi = logpi_gauss, n_iter = n_iter, h = 2.5, x_curr = initial)
mc_l <- RWM(logpi = logpi_laplace, n_iter = n_iter, h = 2.5, x_curr = initial)
mc_c <- RWM(logpi = logpi_cauchy, n_iter = n_iter, h = 5, x_curr = initial)

cat(
  "Acceptance rate for:\n\tGaussian target distribution:", mc_g$a_rate,
  "\n\tLaplace target distribution", mc_l$a_rate,
  "\n\tCauchy target distribution", mc_c$a_rate
)
# The Cauchy distribution has much heavier tails than the normal distribution.
# So a distant points from the current state may still have relatively high density,
# resulting in a high probability of accepting the proposed move.

limits <- range(mc_g$x_store, mc_l$x_store, mc_c$x_store)

plot(1:n_iter, mc_g$x_store, type = "l", xlab = "t", ylab = "X(t)", ylim = limits)
lines(1:n_iter, mc_l$x_store, type = "l", col = "blue")
lines(1:n_iter, mc_c$x_store, type = "l", col = "purple")
legend("bottom",
  legend = c("Gaussian", "Laplace", "Cauchy"),
  col = c("black", "blue", "purple"), horiz = TRUE, lty = 1, lwd = 2, cex = 0.7, bty = "n"
)


# Cold start
initial <- 100

mc_g_cold <- RWM(logpi = logpi_gauss, n_iter = n_iter, h = 2.5, x_curr = initial)
mc_l_cold <- RWM(logpi = logpi_laplace, n_iter = n_iter, h = 2.5, x_curr = initial)
mc_c_cold <- RWM(logpi = logpi_cauchy, n_iter = n_iter, h = 4, x_curr = initial)

mc_warm_cold_samples <- rbind(
  mc_g$x_store,
  mc_g_cold$x_store,
  mc_l$x_store,
  mc_l_cold$x_store,
  mc_c$x_store,
  mc_c_cold$x_store
)

limits <- range(mc_warm_cold_samples)
limits[1] <- limits[1] - 50
matplot(1:n_iter, t(mc_warm_cold_samples),
  ylim = limits,
  type = "l", xlab = "t", ylab = expression(X[1](t)),
  col = brewer.pal(n = 6, name = "Paired"), lty = 1
)
legend("bottom",
  legend = c(
    "Gaussian (warm start)", "Gaussian (cold start)",
    "Laplace (warm start)", "Laplace (cold start)",
    "Cauchy (warm start)", "Cauchy (cold start)"
  ),
  ncol = 3, col = brewer.pal(n = 6, name = "Paired"), lty = 1, lwd = 2, cex = 0.7, bty = "n"
)

# The Gaussian and Laplace distributions are more robust to a cold start. This is because
# even if the sampler starts far from the mean, it will more likely to accept proposals
# that are closer to the center of the distribution due to higher probability density.
# Proposals of large jumps in the extreme are unlikely since the probability density
# decreases rapidly away from the mean.



### HETEROGENEOUS TARGETS
library(mvtnorm)

# Function to calculate the logarithm of the multivariate Gaussian pdf at `x`
logpi_het <- function(x, location, scale) {
  sum(dmvnorm(x, mean = location, sigma = scale, log = TRUE))
}


# Bi-variate Gaussian distribution with different marginal variances
n_iter <- 5000
initial <- 0

mc_h <- RWM(
  logpi = logpi_het, n_iter = n_iter, h = 5.2, x_curr = rep(initial, 2),
  location = rep(0, 2), scale = diag(c(1000, 1))
)
mc_h$a_rate

par(mfrow = c(2, 1))
plot(mc_h$x_store[1, ], type = "l", xlab = "t", ylab = expression(X[1](t)))
plot(mc_h$x_store[2, ], type = "l", xlab = "t", ylab = expression(X[2](t)))


# Bi-variate Gaussian distribution with different marginal variances with
# dependence between components
n_iter <- 1000

mc_corr0.1 <- RWM(
  logpi = logpi_het, n_iter = n_iter, h = 2.5, x_curr = rep(initial, 2),
  location = rep(0, 2), scale = matrix(c(1, 0.1, 0.1, 1), ncol = 2)
)
mc_corr0.1$a_rate

mc_corr0.9 <- RWM(
  logpi = logpi_het, n_iter = n_iter, h = 1.3, x_curr = rep(initial, 2),
  location = rep(0, 2), scale = matrix(c(1, 0.9, 0.9, 1), ncol = 2)
)
mc_corr0.9$a_rate

colors <- brewer.pal(3, "Accent")

par(
  mfrow = c(2, 1),
  mar = c(4, 4, 2, 1),
  oma = c(2, 2, 2, 2),
  xpd = NA
)

limits <- range(mc_corr0.1$x_store, mc_corr0.9$x_store)

plot(mc_corr0.1$x_store[1, ],
  type = "l", ylim = limits,
  xlab = "t", ylab = expression(X[1](t)), col = colors[1]
)
lines(mc_corr0.9$x_store[1, ], col = colors[2])

legend("top",
  inset = c(1, -0.25), legend = c(expression(rho == 0.1), expression(rho == 0.9)),
  horiz = TRUE, col = colors[1:2], lty = 1, lwd = 2, bty = "n"
)

plot(mc_corr0.1$x_store[2, ],
  type = "l", ylim = limits,
  xlab = "t", ylab = expression(X[2](t)), col = colors[1]
)
lines(mc_corr0.9$x_store[2, ], col = colors[2])
