using FactCheck
using TimeModels

include("kalman_filter.jl")
include("parameter_estimation.jl")
include("arima.jl")
include("garch.jl")
include("diagnostic_tests.jl")

exitstatus()
