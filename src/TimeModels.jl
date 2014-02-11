using Stats, DataFrames, Datetime, TimeSeries, Winston 

module TimeModels

using Stats, DataFrames, Datetime, TimeSeries, Winston  

export 
       @timemodels

include("arima.jl")
include("garch.jl")
include("Kalman.jl")
include("../test/testmacro.jl")

end 

