module TimeModels

if VERSION < v"0.4-"
  using Dates
else
  using Base.Dates
end

using Distributions
using StatsBase
using TimeSeries
using NLopt
using Optim
using Compat

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
  predict,
  # diagnostic tests exports
  jbtest

include("Kalman.jl")
include("ARIMA.jl")
include("GARCH.jl")
include("diagnostic_tests.jl")

end
