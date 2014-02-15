# Julia GARCH package
# Copyright 2013 Andrey Kolev
# Distributed under MIT license (see LICENSE.md)

type GarchFit
  data::Vector
  params::Vector
  llh::Float64
  status::Symbol
  converged::Bool
  sigma::Vector
  hessian::Array{Float64,2}
  cvar::Array{Float64,2}
  secoef::Vector
  tval::Vector
end

function Base.show(io::IO ,fit::GarchFit)
  pnorm(x) = 0.5*(1+erf(x/sqrt(2)))
  prt(x) = 2*(1-pnorm(abs(x)))
  @printf io "Fitted garch model \n"
  @printf io " * Coefficient(s): \tomega \t\talpha \t\tbeta\n"
  @printf io "   \t\t\t%f\t%f\t%f\n" fit.params[1] fit.params[2] fit.params[3]
  @printf io " * Log Likelihood: %f\n" fit.llh
  @printf io " * Converged: %s\n" fit.converged
  @printf io " * Solver status: %s\n\n" fit.status
  println(io," * Standardised Residuals Tests:")
  println(io,"   \t\t\t\tStatistic\tp-Value")
  jbstat,jbp = jbtest(fit.data./fit.sigma);
  @printf io "   Jarque-Bera Test\t\U1D6D8\u00B2\t%.6f\t%.6f\n\n" jbstat jbp
  println(io," * Error Analysis:")
  println(io,"   \t\tEstimate\t\Std.Error\tt value \tPr(>|t|)")
  @printf io "   omega\t%f\t%f\t%f\t%f\n" fit.params[1] fit.secoef[1] fit.tval[1] prt(fit.tval[1])
  @printf io "   alpha\t%f\t%f\t%f\t%f\n" fit.params[2] fit.secoef[2] fit.tval[2] prt(fit.tval[2])
  @printf io "   beta \t%f\t%f\t%f\t%f\n"  fit.params[3] fit.secoef[3] fit.tval[3] prt(fit.tval[3])
end

function cdHessian(par,LLH)
  eps = 1e-4 * par
  n = length(par)
  H = zeros(n,n)
  for(i = 1:n)
    for(j = 1:n)
      x1 = copy(par) 
      x1[i] += eps[i]
      x1[j] += eps[j] 
      x2 = copy(par)
      x2[i] += eps[i]
      x2[j] -= eps[j]
      x3 = copy(par)
      x3[i] -= eps[i]
      x3[j] += eps[j]
      x4 = copy(par)
      x4[i] -= eps[i]
      x4[j] -= eps[j]
      H[i,j] = (LLH(x1)-LLH(x2)-LLH(x3)+LLH(x4)) / (4.*eps[i]*eps[j])
    end
  end
  H
end

function garchLLH(rets::Vector,x::Vector)
  rets2   = rets.^2;
  T = length(rets); 
  ht = zeros(T);
  omega,alpha,beta = x;
  ht[1] = sum(rets2)/T;
  for i=2:T
    ht[i] = omega + alpha*rets2[i-1] + beta * ht[i-1];
  end
  -0.5*(T-1)*log(2*pi)-0.5*sum( log(ht) + (rets./sqrt(ht)).^2 );
end

function predict(fit::GarchFit)
 omega, alpha, beta = fit.params;
 rets = fit.data
 rets2   = rets.^2;
 T = length(rets); 
 ht    = zeros(T);
 ht[1] = sum(rets2)/T;
 for i=2:T
    ht[i] = omega + alpha*rets2[i-1] + beta * ht[i-1];
 end
 sqrt(omega + alpha*rets2[end] + beta*ht[end]);
end

function garchFit(data::Vector)
  rets = data
  rets2   = rets.^2;
  T = length(rets); 
  ht = zeros(T);
  function garchLike(x::Vector, grad::Vector)
    omega,alpha,beta = x;
    ht[1] = sum(rets2)/T;
    for i=2:T
      ht[i] = omega + alpha*rets2[i-1] + beta * ht[i-1];
    end
    sum( log(ht) + (rets./sqrt(ht)).^2 );
  end
  opt = Opt(:LN_SBPLX,3)
  lower_bounds!(opt,[1e-10, 0.0, 0.0])
  upper_bounds!(opt,[1; 0.3; 0.99])
  min_objective!(opt, garchLike)
  (minf,minx,ret) = optimize(opt, [1e-5, 0.09, 0.89])
  converged = minx[1]>0 && all(minx[2:3].>=0) && sum(minx[2:3])<1.0
  H = cdHessian(minx,x->garchLLH(rets,x))
  cvar = -inv(H)
  secoef = sqrt(diag(cvar))
  tval = minx./secoef
  out = GarchFit(data, minx, -0.5*(T-1)*log(2*pi)-0.5*minf, ret, converged, sqrt(ht),H,cvar,secoef,tval)
end

# function garchPkgTest()
#   println("Running GARCH package test...")
#   try
#     include(Pkg.dir("GARCH", "test","GARCHtest.jl"))
#     println("All tests passed!")
#   catch err
#     throw(err)
#   end
# end
