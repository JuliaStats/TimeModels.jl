
println("Testing Kalman.jl")
include("../src/Kalman.jl")
# include("./src/Kalman.jl")

srand(1)

F = diagm([1.0])
V = diagm([2.0])
G = reshape([1, 2, -0.5], 3, 1)
W = diagm([8.0, 2.5, 4.0])
x0 = randn(1)
P0 = diagm([1e7])
mod1 = Kalman.StateSpaceModel(F, V, G, W, x0, P0)

# Test simulating 
println("Simulating series")
x, y = Kalman.simulate(mod1, 100)
@time x, y = Kalman.simulate(mod1, 100)
t = [1:size(x, 1)]


# Test filtering
println("Filtering")
filt = Kalman.kalman_filter(y, mod1)
@time filt = Kalman.kalman_filter(y, mod1)
println("Log likelihood:")
println(filt.loglik)

# Test smoothing
println("Smoothing")
smooth = Kalman.kalman_smooth(y, mod1)
@time smooth = Kalman.kalman_smooth(y, mod1)

# println("Plotting")
# p = FramedPlot()
# for i=1:3
# 	add(p, Curve(t, y[:, i], "color", "grey"))
# end
# add(p, Curve(t, filt.filtered, "type", "dashed"))
# add(p, Curve(t, smooth.smoothed, "color", "blue"))
# # add(p, Curve(t, x[:, 1], "color", "blue"))
# # add(p, Curve(t, filt.filtered[:, 1], "color", "green"))
# # add(p, Curve(t, smooth.smoothed[:, 1], "color", "red"))
# Winston.display(p)

println("\n\nPassed all tests.")
