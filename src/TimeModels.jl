module TimeModels

using Base.Dates
using Distributions
using StatsBase
using TimeSeries
using Optim

import Base: show

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
  # diagnostic tests exports
  jbtest

# Core functionality
include("statespacemodel.jl")
include("kalman_filter.jl")
include("kalman_smooth.jl")
include("parameter_estimation.jl")

# Model specifications
include("ARIMA.jl")
include("GARCH.jl")

# Tests
include("diagnostic_tests.jl")

end
