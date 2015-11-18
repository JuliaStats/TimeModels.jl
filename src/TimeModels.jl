module TimeModels

using Base.Dates
using Distributions
using StatsBase
using TimeSeries 
using Optim

import Base: show

export
  # General state space model 
  StateSpaceModel, simulate,

  #Kalman
  kalman_filter, KalmanFiltered,
  kalman_smooth, KalmanSmoothed, KalmanSmoothedMinimal,

  # Parameter fitting
  ParametrizedMatrix,
  parametrize_full, parametrize_diag, parametrize_none, 
  ParametrizedSSM, SSMParameters, fit, 

  # ARIMA exports
  arima_statespace,
  arima,

  # GARCH exports
  garchFit,
  predict,

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
