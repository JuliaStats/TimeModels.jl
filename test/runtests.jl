# adding using TimeModels to this file is odd, but it's a trick to get travis to pass

using Base.Test
using TimeModels

include("kalman_filter.jl")
include("parameter_estimation.jl")
