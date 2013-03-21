using Winston

require("kalman.jl")

srand(1)

F = diagm([1.0])
V = diagm([2.0])
G = reshape([1, 2, -0.5], 3, 1)
W = diagm([8.0, 2.5, 4.0])
x0 = rnorm(1)
P0 = diagm([1e7])
mod1 = StateSpaceModel(F, V, G, W, x0, P0)

check_dimensions(mod1)
# Test that the thing can handle single- and multidimensional series

# Test simulating 
x, y = simulate_statespace(100, mod1)
t = [1:size(x, 1)]


# Test filtering

filt = kalman_filter(y, mod1)

# Test smoothing

smooth = kalman_smooth(y, mod1)


p = FramedPlot()
for i=1:3
	add(p, Curve(t, y[:, i], "color", "grey"))
end
add(p, Curve(t, filt.filtered, "type", "dashed"))
add(p, Curve(t, smooth.smoothed, "color", "blue"))
# add(p, Curve(t, x[:, 1], "color", "blue"))
# add(p, Curve(t, filt.filtered[:, 1], "color", "green"))
# add(p, Curve(t, smooth.smoothed[:, 1], "color", "red"))
file(p, "series.png")

println("\n\nPassed all tests.")
