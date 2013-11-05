using DataFrames, Datetime, TimeSeries  #, RMath, Winston

module TimeModels

using DataFrames, Datetime, TimeSeries  #, RMath, Winston

export acf, 
       arima, 
       garch, 
## testing
       @timemodels


include("acf.jl")
include("arima.jl")
include("garch.jl")
include("testtimemodels.jl")

end 

