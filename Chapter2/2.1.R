# 个人部分
# 2.1
# ------------------------------------------------------------------------------
# a.

# Define log likelihood function function
L_theta <- function(x_obs, x_from, x_to, x_by) {
  # Plot the log likelihood function of Cauchy(theta, 1).
  #
  # Args:
  #   x_obs: Observations of x.
  #   x_from, x_to, x_by: Specify the x-axis.
  
  # Initial values
  n <- length(x_obs)
  theta <- seq(x_from, x_to, x_by)
  y <- vector("numeric")
  
  # Calculate y
  for (i in 1:length(theta)) {
    one_theta <- theta[i]
    L <- - n * log(pi) - sum(log((x_obs - one_theta) ** 2 + 1))
    y[i] <- L
  }
  
  # Plot
  plot(theta, y, xlab = expression(theta), 
       ylab = expression(paste("L(", theta, ")")),
       cex = 0.001, pch = 19)
}


# Newton's method
L_theta_prime <- function(x_obs, theta) {
  # Calculate L' (Cauchy(theta, 1)).
  #
  # Args:
  #   x_obs: Observations of x.
  #   theta: Parameter of Cauchy(theta, 1).
  #
  # Returns:
  #   Value of L'.
  
  return(2 * sum((x_obs - theta) / ((x_obs - theta) ** 2 + 1)))
}

L_theta_2prime <- function(x_obs, theta) {
  # Calculate L''.
  #
  # Args:
  #   x_obs: Observations of x.
  #   theta: Parameter of Cauchy(theta, 1).
  #
  # Returns:
  #   Value of L''.
  
  return(2 * sum(((x_obs - theta) ** 2 - 1) / ((x_obs - theta) ** 2 + 1) ** 2))
}

L_theta_prime_Normal <- function(x_obs, theta) {
  # Calculate L' (Normal(theta, 1)).
  #
  # Args:
  #   x_obs: Observations of x.
  #   theta: Parameter of Cauchy(theta, 1).
  #
  # Returns:
  #   Value of L'.
  
  return(sum(x_obs - theta))
}

L_theta_2prime_Normal <- function(x_obs, theta) {
  # Calculate L'' (Normal(theta, 1)).
  #
  # Args:
  #   x_obs: Observations of x.
  #   theta: Parameter of Cauchy(theta, 1).
  #
  # Returns:
  #   Value of L''.
  
  return(-length(x_obs))
}
  
Newtons_method <- function(theta, x_obs, epsilon, max_iteration, type) {
  # Estimate theta by Newton's method. Use absolute convergence criterion.
  #
  # Args:
  #   theta: initial theta.
  #   x_obs: Observations of x.
  #   epsilon: Tolerable imprecision.
  #   max_iteration: Max iterations.
  #   type: Cauchy or Normal
  #
  # Returns:
  #   Estimated theta, run time, stopped iteration and estimated values at each iteration.
  
  # Initiate values
  start_time <- as.numeric(Sys.time()) * 1000
  theta_previous <- theta + epsilon + 1
  theta_current <- theta
  iteration <- 0
  estimates <- vector("numeric")
  
  # Iterate
  while ((abs(theta_current - theta_previous) > epsilon) && (iteration < max_iteration)) {
    theta_previous <- theta_current
    if (type == "Cauchy") {
      theta_current <- theta_previous - L_theta_prime(x_obs, theta_previous) / L_theta_2prime(x_obs, theta_previous)
    } else if (type == "Normal") {
      theta_current <- theta_previous - L_theta_prime_Normal(x_obs, theta_previous) / L_theta_2prime_Normal(x_obs, theta_previous)
    } else {
      print("Wrong type!")
      return(1)
    }
    estimates[length(estimates) + 1] <- theta_current
    iteration <- iteration + 1
  }
  
  # Warning: not converge
  if (iteration == max_iteration) {
    print(paste("Warning: Fail to converge, ", theta, "."))
  }
  
  # Run time
  end_time <- as.numeric(Sys.time()) * 1000
  duration <- end_time - start_time
  
  # Output
  return(list("theta_hat" = theta_current, "run_time" = duration, "stop" = iteration, "estimates" = estimates))
}


# Main
x_obs <- c(1.77, -0.23, 2.76, 3.80, 3.47, 56.75, -1.34, 4.24, -2.44, 3.29, 
           3.71, -2.40, 4.53, -0.07, -1.05, -13.87, -2.53, -1.75, 0.27, 43.21)
L_theta(x_obs, -10, 10, 0.001) # Graph

epsilon <- 10 ** (-6)
max_iteration <- 100
theta_0 <- c(-11, -1, 0, 1.5, 4, 4.7, 7, 8, 38)
theta_hat <- vector("numeric")
for (i in 1:length(theta_0)) {
  theta <- theta_0[i]
  theta_hat[i] <- Newtons_method(theta, x_obs, epsilon, max_iteration, "Cauchy")$theta_hat
} # Estimate theta from different start points
Newton_result <- data.frame(theta_0, theta_hat)
print(Newton_result)

x_obs_mean <- mean(x_obs) # Mean of the data
x_obs_mean_theta <- Newtons_method(x_obs_mean, x_obs, epsilon, max_iteration, "Cauchy")$theta_hat
#-------------------------------------------------------------------------------
# b.

# Bisection method
bisection_method <- function(a_0, b_0, x_obs, epsilon, max_iteration, type) {
  # Estimate theta by bisection method. Use absolute convergence criterion.
  # 
  # Args:
  #   a_0, b_0: Initial values [a, b].
  #   x_obs: Observations of x.
  #   epsilon: Tolerable imprecision.
  #   max_iteration: Max iterations.
  #   type: Cauchy or Normal
  #
  # Returns:
  #   Estimated theta, run time, stopped iteration and estimated values at each iteration.
  
  # Initiate values
  start_time <- as.numeric(Sys.time()) * 1000
  a_t <- a_0
  b_t <- b_0
  x_current <- a_0 + (b_0 - a_0) / 2
  x_previous <- x_current + epsilon + 1
  iteration <- 0
  estimates <- vector("numeric")
  
  # Iterate
  while ((abs(x_current - x_previous) > epsilon) && (iteration < max_iteration)) {
    x_previous <- x_current
    if (type == "Cauchy") {
      if (L_theta_prime(x_obs, a_t) * L_theta_prime(x_obs, x_previous) <= 0) {
        b_t <- x_previous
      } else {
        a_t <- x_previous
      }
    } else if (type == "Normal") {
      if (L_theta_prime_Normal(x_obs, a_t) * L_theta_prime_Normal(x_obs, x_previous) <= 0) {
        b_t <- x_previous
      } else {
        a_t <- x_previous
      }
    } else {
      print("Wrong type!")
      return(1)
    }
    x_current <- a_t + (b_t - a_t) / 2
    iteration <- iteration + 1
    estimates[length(estimates) + 1] <- x_current
  }
  
  # Warning: not converge
  if (iteration == max_iteration) {
    print("Warning: Fail to converge.")
  }
  
  # Run time
  end_time <- as.numeric(Sys.time()) * 1000
  duration <- end_time - start_time
  
  # Output
  return(list("theta_hat" = x_current, "run_time" = duration, "stop" = iteration, "estimates" = estimates))
}


# Main
a_0 <- -1
b_0 <- 1
bisection_result <- bisection_method(a_0, b_0, x_obs, epsilon, max_iteration, "Cauchy")$theta_hat

a_0_local <- c(-1, -1, -0.2, -0.19)
b_0_local <- c(4.42, 4.43, 1, 1)
bisection_result_local <- vector("numeric")
for (i in 1:length(a_0_local)) {
  bisection_result_local[i] <- bisection_method(a_0_local[i], b_0_local[i], x_obs, epsilon, max_iteration, "Cauchy")$theta_hat
}
print(bisection_result_local)
#-------------------------------------------------------------------------------
# c.

# Fixed-point iteration
fixed_point_iteration <- function(alpha, x_0, x_obs, epsilon, max_iteration, type) {
  # Estimate theta by fixed-point iteration. Use absolute convergence criterion.
  #
  # Args:
  #   alpha: Parameter alpha.
  #   x_0: Initial theta.
  #   x_obs: Observations of x.
  #   epsilon: Tolerable imprecision.
  #   max_iteration: Max iterations.
  #   type: Cauchy or Normal
  #
  # Returns:
  #   Estimated theta, run time, stopped iteration and estimated values at each iteration.
  
  # Initiate values
  start_time <- as.numeric(Sys.time()) * 1000
  x_previous <- x_0 + epsilon + 1
  x_current <- x_0
  iteration <- 0
  estimates <- vector("numeric")
  
  # Iteration
  while ((abs(x_current - x_previous) > epsilon) && (iteration < max_iteration)) {
    x_previous <- x_current
    if (type == "Cauchy") {
      x_current <- alpha * L_theta_prime(x_obs, x_previous) + x_previous
    } else if (type == "Normal") {
      x_current <- alpha * L_theta_prime_Normal(x_obs, x_previous) + x_previous
    } else {
      print("Wrong type!")
      return(1)
    }
    iteration <- iteration + 1
    estimates[length(estimates) + 1] <- x_current
  }
  
  # Warning: not converge
  if (iteration == max_iteration) {
    print(paste("Warning: Fail to converge, alpha: ", alpha, ", x_0: ", x_0, "."))
  }
  
  # Run time
  end_time <- as.numeric(Sys.time()) * 1000
  duration <- end_time - start_time
  
  # Output
  return(list("theta_hat" = x_current, "run_time" = duration, "stop" = iteration, "estimates" = estimates))
}


# Main
x_0 <- -1
alpha <- c(1, 0.64, 0.25)
fixed_point_result <- vector("numeric")
max_iteration <- 500
for (i in 1:length(alpha)) {
  fixed_point_result[i] <- fixed_point_iteration(alpha[i], x_0, x_obs, epsilon, max_iteration, "Cauchy")$theta_hat
}
print(fixed_point_result)

x_0_trials <- seq(-5, 5, 0.01) # Try other values
alpha_trials <- seq(0.01, 1, 0.01)
fixed_point_trails <- vector("numeric")
for (i in 1:length(x_0_trials)) {
  for (j in 1:length(alpha_trials)) {
    fixed_point_trails[length(fixed_point_trails) + 1] <- fixed_point_iteration(alpha_trials[j], x_0_trials[i], x_obs, epsilon, max_iteration, "Cauchy")$theta_hat
  }
}
fixed_point_trails_matrix <- matrix(fixed_point_trails, length(x_0_trials), length(alpha_trials), byrow = TRUE)
filled.contour(x_0_trials, alpha_trials, fixed_point_trails_matrix, color.palette = terrain.colors, 
               xlab = expression(x[0]), ylab = expression(alpha)) # Visualization
#-------------------------------------------------------------------------------
# d.

# Secant method
secant_method <- function(x_0, x_neg_1, x_obs, epsilon, max_iteration, type) {
  # Estimate theta by fixed-point iteration. Use absolute convergence criterion.
  #
  # Args:
  #   x_0, x_neg_1: Initial theta.
  #   x_obs: Observations of x.
  #   epsilon: Tolerable imprecision.
  #   max_iteration: Max iterations.
  #   type: Cauchy or Normal
  #
  # Returns:
  #   Estimated theta, run time, stopped iteration and estimated values at each iteration.
  
  # Initiate values
  start_time <- as.numeric(Sys.time()) * 1000
  x_t_min_one <- x_neg_1
  x_t <- x_0
  iteration <- 0
  estimates <- vector("numeric")
  
  # Iterate
  while ((abs(x_t - x_t_min_one) > epsilon) && (iteration < max_iteration)) {
    if (type == "Cauchy") {
      x_t_plus_one <- x_t - (L_theta_prime(x_obs, x_t) * (x_t - x_t_min_one)) / (L_theta_prime(x_obs, x_t) - L_theta_prime(x_obs, x_t_min_one))
    } else if (type == "Normal") {
      x_t_plus_one <- x_t - (L_theta_prime_Normal(x_obs, x_t) * (x_t - x_t_min_one)) / (L_theta_prime_Normal(x_obs, x_t) - L_theta_prime_Normal(x_obs, x_t_min_one))
    } else {
      print("Wrong type!")
      return(1)
    }
    x_t_min_one <- x_t 
    x_t <- x_t_plus_one
    iteration <- iteration + 1
    estimates[length(estimates) + 1] <- x_t
  }
  
  # Warning: not converge
  if (iteration == max_iteration) {
    print(paste("Warning: Fail to converge, x_0: ", x_0, ", x_neg_1: ", x_neg_1, "."))
  }
  
  # Run time
  end_time <- as.numeric(Sys.time()) * 1000
  duration <- end_time - start_time
  
  # Output
  return(list("theta_hat" = x_t, "run_time" = duration, "stop" = iteration, "estimates" = estimates))
}


# Main
max_iteration <- 100
x_neg_1 <- -2
x_0 <- -1
secant_result <- secant_method(x_0, x_neg_1, x_obs, epsilon, max_iteration, "Cauchy")$theta_hat
print(secant_result)

x_neg_1 <- -3
x_0 <- 3
secant_result <- secant_method(x_0, x_neg_1, x_obs, epsilon, max_iteration, "Cauchy")$theta_hat
print(secant_result)

x_neg_1 <- seq(-5, 0, 0.01) # Try other values
x_0 <- seq(0, 5, 0.01)
secant_trials <- vector("numeric")
for (i in 1:length(x_neg_1)) {
  for (j in 1:length(x_0)) {
    secant_trials[length(secant_trials) + 1] <- secant_method(x_0[j], x_neg_1[i], x_obs, epsilon, max_iteration, "Cauchy")$theta_hat
  }
}
secant_trials_matrix <- matrix(secant_trials, length(x_neg_1), length(x_0), byrow = TRUE)
filled.contour(x_neg_1, x_0, secant_trials_matrix, color.palette = terrain.colors, 
               xlab = expression(theta^{(0)}), ylab = expression(theta^{(1)})) # Visualization
#-------------------------------------------------------------------------------
# e.

# Speed: Cauchy(theta, 1)
max_iteration <- 100
Newton_theta_0 <- -1
Newton_Cauchy <- Newtons_method(Newton_theta_0, x_obs, epsilon, max_iteration, "Cauchy")
print(Newton_Cauchy)

bisection_a_0 <- -1
bisection_b_0 <- 1
bisection_Cauchy <- bisection_method(bisection_a_0, bisection_b_0, x_obs, epsilon, max_iteration, "Cauchy")
print(bisection_Cauchy)

fixed_point_alpha <- 0.25
fixed_point_theta_0 <- -1
fixed_point_Cauchy <- fixed_point_iteration(fixed_point_alpha, fixed_point_theta_0, x_obs, epsilon, max_iteration, "Cauchy")
print(fixed_point_Cauchy)

secant_theta_0 <- -2
secant_theta_neg_1 <- -1
secant_Cauchy <- secant_method(secant_theta_0, secant_theta_neg_1, x_obs, epsilon, max_iteration, "Cauchy")
print(secant_Cauchy)

plot(1:length(bisection_Cauchy$estimates), bisection_Cauchy$estimates, type = "l", 
     col = "red", xlab = "Iteration", ylab = expression(hat(theta)))
lines(1:length(Newton_Cauchy$estimates), Newton_Cauchy$estimates, col = "blue")
lines(1:length(fixed_point_Cauchy$estimates), fixed_point_Cauchy$estimates, col = "green")
lines(1:length(secant_Cauchy$estimates), secant_Cauchy$estimates, col = "orange")
legend("bottomright", inset = 0.01, c("Newton's", "bisection", "fixed-point", "secant"), lty = c(1, 1, 1, 1),
       col = c("blue", "red", "green", "orange"), cex = 0.7, text.width = 1)

# Stability: Cauchy(theta, 1)
theta_0 <- seq(-5, 5, 0.01)
Newton_trials <- vector("numeric")
for (i in 1:length(theta_0)) {
  Newton_trials[length(Newton_trials) + 1] <- Newtons_method(theta_0[i], x_obs, epsilon, max_iteration, "Cauchy")$theta_hat
}
plot(theta_0, Newton_trials, type = "b", pch = 19, cex = 0.1,
     xlab = expression(theta[0]), ylab = expression(hat(theta)))

a_0 <- seq(-5, 0, 0.01)
b_0 <- seq(0, 5, 0.01)
bisection_trials <- vector("numeric")
for (i in 1:length(a_0)) {
  for (j in 1:length(b_0)) {
    bisection_trials[length(bisection_trials) + 1] <- bisection_method(a_0[i], b_0[j], x_obs, epsilon, max_iteration, "Cauchy")$theta_hat
  }
}
bisection_trials_matrix <- matrix(bisection_trials, length(a_0), length(b_0), byrow = TRUE)
filled.contour(a_0, b_0, bisection_trials_matrix, color.palette = terrain.colors, 
               xlab = expression(a[0]), ylab = expression(b[0])) # Visualization

# Speed: Normal(theta, 1)
set.seed(1234)
x_obs <- rnorm(20, mean = 1, sd = 1)

Newton_theta_0 <- 3
Newton_Normal <- Newtons_method(Newton_theta_0, x_obs, epsilon, max_iteration, "Normal")
print(Newton_Normal)

bisection_a_0 <- -1
bisection_b_0 <- 1
bisection_Normal <- bisection_method(bisection_a_0, bisection_b_0, x_obs, epsilon, max_iteration, "Normal")
print(bisection_Normal)

fixed_point_alpha <- 0.05
fixed_point_theta_0 <- 2
fixed_point_Normal <- fixed_point_iteration(fixed_point_alpha, fixed_point_theta_0, x_obs, epsilon, max_iteration, "Normal")
print(fixed_point_Normal)

secant_theta_0 <- -2
secant_theta_neg_1 <- -1
secant_Normal <- secant_method(secant_theta_0, secant_theta_neg_1, x_obs, epsilon, max_iteration, "Normal")
print(secant_Normal)

plot(1:length(bisection_Normal$estimates), bisection_Normal$estimates, type = "l", 
     col = "red", xlab = "Iteration", ylab = expression(hat(theta)))
lines(1:length(Newton_Normal$estimates), Newton_Normal$estimates, col = "blue")
lines(1:length(fixed_point_Normal$estimates), fixed_point_Normal$estimates, col = "green")
lines(1:length(secant_Normal$estimates), secant_Normal$estimates, col = "orange")
legend("bottomright", inset = 0.01, c("Newton's", "bisection", "fixed-point", "secant"), lty = c(1, 1, 1, 1),
       col = c("blue", "red", "green", "orange"), cex = 0.5, text.width = 1)

# Stability: Normal(theta, 1)
theta_0 <- seq(-5, 5, 0.01)
Newton_trials <- vector("numeric")
for (i in 1:length(theta_0)) {
  Newton_trials[length(Newton_trials) + 1] <- Newtons_method(theta_0[i], x_obs, epsilon, max_iteration, "Normal")$theta_hat
}
plot(theta_0, Newton_trials, type = "b", pch = 19, cex = 0.1,
     xlab = expression(theta[0]), ylab = expression(hat(theta)))

a_0 <- seq(-5, 0, 0.01)
b_0 <- seq(0, 5, 0.01)
bisection_trials <- vector("numeric")
for (i in 1:length(a_0)) {
  for (j in 1:length(b_0)) {
    bisection_trials[length(bisection_trials) + 1] <- bisection_method(a_0[i], b_0[j], x_obs, epsilon, max_iteration, "Normal")$theta_hat
  }
}
bisection_trials_matrix <- matrix(bisection_trials, length(a_0), length(b_0), byrow = TRUE)
filled.contour(a_0, b_0, bisection_trials_matrix, color.palette = terrain.colors, 
               xlab = expression(a[0]), ylab = expression(b[0])) # Visualization

x_0_trials <- seq(-5, 5, 0.01)
alpha_trials <- seq(0.01, 1, 0.01)
fixed_point_trails <- vector("numeric")
for (i in 1:length(x_0_trials)) {
  for (j in 1:length(alpha_trials)) {
    fixed_point_trails[length(fixed_point_trails) + 1] <- fixed_point_iteration(alpha_trials[j], x_0_trials[i], x_obs, epsilon, max_iteration, "Normal")$theta_hat
  }
}
fixed_point_trails_matrix <- matrix(fixed_point_trails, length(x_0_trials), length(alpha_trials), byrow = TRUE)
filled.contour(x_0_trials, alpha_trials, fixed_point_trails_matrix, color.palette = terrain.colors, 
               xlab = expression(x[0]), ylab = expression(alpha)) # Visualization

x_neg_1 <- seq(-5, 0, 0.01)
x_0 <- seq(0, 5, 0.01)
secant_trials <- vector("numeric")
for (i in 1:length(x_neg_1)) {
  for (j in 1:length(x_0)) {
    secant_trials[length(secant_trials) + 1] <- secant_method(x_0[j], x_neg_1[i], x_obs, epsilon, max_iteration, "Normal")$theta_hat
  }
}
secant_trials_matrix <- matrix(secant_trials, length(x_neg_1), length(x_0), byrow = TRUE)
filled.contour(x_neg_1, x_0, secant_trials_matrix, color.palette = terrain.colors, 
               xlab = expression(theta^{(0)}), ylab = expression(theta^{(1)})) # Visualization
