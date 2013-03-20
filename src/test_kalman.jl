using Winston

require("kalman.jl")

srand(1)

F = diagm([1.0, 1.0])
V = diagm([2.0, 1.0])
G = diagm([3.0 5.0])
W = diagm([8.0, 2.5])
x0 = rnorm(2)
P0 = ones(2, 2) * 1e7
mod1 = StateSpaceModel(F, V, G, W, x0, P0)

check_dimensions(mod1)
# Test that the thing can handle single- and multidimensional series

# Test simulating 
x, y = simulate_statespace(100, mod1)
t = [1:size(x, 1)]


# Test filtering

x_est = KalmanFilter(y, mod1)

# Test smoothing


p = FramedPlot()
add(p, Curve(t, x[:, 1], "color", "red"))
add(p, Curve(t, x[:, 2], "color", "blue"))
add(p, Curve(t, x_est[:, 1], "color", "red", "type", "dotted"))
add(p, Curve(t, x_est[:, 2], "color", "blue", "type", "dotted"))
file(p, "series.png")

println("\n\nPassed all tests.")
