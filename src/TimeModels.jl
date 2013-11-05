using DataFrames, Datetime, TimeSeries, Winston  #, RMath

module TimeModels

using DataFrames, Datetime, TimeSeries, Winston  #, RMath

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

