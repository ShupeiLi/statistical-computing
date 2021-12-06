# Problem 6.1
y = read.table(file.choose(), header = T)
options(digits = 15)
b.0 = c(1.804, 0.165)
s2.0 = 0.015^2
n = 256

# Functions
# Original function, f
f = function(g, s2, b, id, trans1 = FALSE, trans2 = FALSE){
  x = matrix(c(rep(1, 5), 1:5), 5, 2)
  lambda = exp(x %*% b + rep(g, 5))
  y.sub = y[y$subject == id, 3]
  out = (dnorm(g, mean = 0, sd = sqrt(s2)) * ((1:5) %*% (y.sub - lambda)) * 
              prod(dpois(y.sub,lambda)))
  
  # trans1: sqrt(f), trans2:x * sqrt(f)
  if (trans1) {
    return(sqrt(out))
  } else if (trans2) {
    return(g * sqrt(out))
  } else {
    return(out)
  }
}

x = seq(-0.2, 0.2, 0.00001)
outs0 = vector("numeric")
outs1 = vector("numeric")
outs2 = vector("numeric")
for (i in 1:length(x)) {
  outs0[i] = f(x[i], s2.0, b.0, 1)
  outs1[i] = f(x[i], s2.0, b.0, 1, TRUE)
  outs2[i] = f(x[i], s2.0, b.0, 1, FALSE, TRUE)
}

plot(x, outs0, type = "l", xlab = "x", ylab = "f")
plot(x, outs1, type = "l", xlab = "x", ylab = "sqrt(f)")
plot(x, outs2, type = "l", xlab = "x", ylab = "x sqrt(f)")

u = max(outs1)
v_minus = min(outs2)
v_plus = max(outs2)

# Define envelope function
envelope = function(g, u, v_minus, v_plus) {
  check_point1 = v_minus / u
  check_point2 = v_plus / u
  if (g < check_point1) {
    return((v_minus / g)^2)
  } else if (g > check_point2) {
    return((v_plus / g)^2)
  } else {
    return(u^2)
  }
}

env_outs = vector("numeric")
for (i in 1:length(x)) {
  env_outs[i] = envelope(x[i], u, v_minus, v_plus)
}

# Plot the integrand and envelope
plot(x, outs0, type = "l", xlab = "x", ylab = "f")
lines(x, env_outs, col = "red")
