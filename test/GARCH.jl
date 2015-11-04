using FactCheck
using TimeModels
using TimeModels.GARCH

facts("GARCH") do
    context("Consistency with R's fGarch") do
        filename = Pkg.dir("TimeModels", "test", "data", "random process.csv")
        ret = readcsv(filename)[:, 1]
        ret = ret .- mean(ret) #Note: GARCH does not remove the mean, so we have to remove it ourselves.
        fit = fit_GARCH(ret)
        param = [2.469347e-06, 1.142268e-01, 8.691734e-01] #R fGarch garch(1,1) estimated params on the zeroized return.
        @fact fit.params --> roughly(param, atol=1e-3)
    end
end
