using DataFrames, Calendar, UTF16, TimeSeries

module TimeModels

using DataFrames, Calendar, UTF16, TimeSeries

export acf, 
       arima, 
       garch, 
## testing
       @timemodels,


include("acf.jl")
include("arima.jl")
include("garch.jl")
include("testtimemodels.jl")

end 
