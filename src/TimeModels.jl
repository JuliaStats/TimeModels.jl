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

import Base.call
import Base.show
import Base.length
import Base.size
import Base.filter

export

    # Model specification exports
    StateSpaceModel,
    ParametrizedMatrix,
    ParametrizedSSM,
    SSMParameters,
    parametrize_full,
    parametrize_diag,
    parametrize_none,

    # Kalman exports
    KalmanFiltered, 
    KalmanSmoothed,
    simulate,
    smooth,
    fit, 

    # ARIMA exports
    arima_statespace,
    arima,

    # GARCH exports
    garchFit,
    predict,

    # diagnostic tests exports
    jbtest

include("statespacemodel.jl")
include("kalman.jl")
include("parameter_estimation.jl")
include("ARIMA.jl")
include("GARCH.jl")
include("diagnostic_tests.jl")

end 
