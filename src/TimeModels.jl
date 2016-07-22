module TimeModels

using Base.Dates
using Distributions
using StatsBase
using TimeSeries 
using Optim
using NLopt

import Base: show

export
  # General state space model 
  StateSpaceModel, simulate,

  #Kalman
  loglikelihood, kalman_filter, KalmanFiltered,
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
  jbtest,

  # utilities
  em_checkmodel

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

include("utilities.jl")

end 
