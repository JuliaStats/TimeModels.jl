using Winston

require("kalman.jl")

println("Finished loading packages.")

srand(1)

F = diagm([1.0])
V = diagm([2.0])
G = reshape([1, 2, -0.5], 3, 1)
W = diagm([8.0, 2.5, 4.0])
x0 = randn(1)
P0 = diagm([1e7])
mod1 = StateSpaceModel(F, V, G, W, x0, P0)

check_dimensions(mod1)
# Test that the thing can handle single- and multidimensional series

# Test simulating 
println("Simulating series")
@time begin
	x, y = simulate_statespace(100, mod1)
	t = [1:size(x, 1)]
end


# Test filtering
println("Filtering")
@time begin
	filt = kalman_filter(y, mod1)
	println("Log likelihood:")
	println(filt.loglik)
end

# Test smoothing
println("Smoothing")
@time smooth = kalman_smooth(y, mod1)

println("Plotting")
p = FramedPlot()
for i=1:3
	add(p, Curve(t, y[:, i], "color", "grey"))
end
add(p, Curve(t, filt.filtered, "type", "dashed"))
add(p, Curve(t, smooth.smoothed, "color", "blue"))
# add(p, Curve(t, x[:, 1], "color", "blue"))
# add(p, Curve(t, filt.filtered[:, 1], "color", "green"))
# add(p, Curve(t, smooth.smoothed[:, 1], "color", "red"))
Winston.display(p)

println("\n\nPassed all tests.")
