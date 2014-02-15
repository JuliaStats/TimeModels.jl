module TimeModels

using Distributions
using StatsBase
using DataFrames
using Datetime
using TimeSeries 
using NLopt
using Optim

import Base.show

export
  # Kalman exports
  StateSpaceModel,
  KalmanFiltered, 
  KalmanSmoothed,
  simulate,
  kalman_filter,
  kalman_smooth,
  fit, 
  # ARIMA exports
  arima_statespace,
  arima,
  # GARCH exports
  garchFit,
  predict

include("Kalman.jl")
include("ARIMA.jl")
include("GARCH.jl")

end 
