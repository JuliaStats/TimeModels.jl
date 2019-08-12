# Tests the null of normality using the Jarque-Bera test statistic.
function jbtest(x::Vector)
    n = length(x)
    m1 = sum(x)/n
    m2 = sum((x - m1).^2)/n
    m3 = sum((x - m1).^3)/n
    m4 = sum((x - m1).^4)/n
    b1 = (m3/m2^(3/2))^2
    b2 = (m4/m2^2)
    statistic = n * b1/6 + n*(b2 - 3)^2/24
    d = Chisq(2.)
    pvalue = 1.0 - cdf(d,statistic)
    statistic, pvalue
end
