using Base.Test
using GARCH

filename = Pkg.dir("GARCH", "test","data","SPY.csv")
close = readcsv(filename,Float64)[:,2]
ret = diff(log(close))
ret = ret - mean(ret)
fit = garchFit(ret)
param = [2.469347e-06, 1.142268e-01, 8.691734e-01] #R fGarch garch(1,1) estimated params
@test_approx_eq_eps(fit.params,param,1e-3)


