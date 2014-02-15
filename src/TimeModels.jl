using StatsBase, DataFrames, Datetime, TimeSeries

module TimeModels

using StatsBase, DataFrames, Datetime, TimeSeries 

export 
  @timemodels

include("Kalman.jl")
include("ARIMA.jl")
include("GARCH.jl")
include("../test/testmacro.jl")

end 
