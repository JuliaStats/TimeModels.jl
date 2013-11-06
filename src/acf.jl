acf = function(x::Array, n::Int)

  acf_array = ones(n)
  [acf_array[i] = cor(x[1:end-(i-1)], x[i:end]) for i in 2:n]
  
## I think this is causing the travis-ci build to fail
##  #plot with Winston
##  y = linspace(0,n-1,n)
##  x = acf_array
##  zp = 0.05(ones(n))
##  zn = -0.05(ones(n))
##
##  plot(y, x, "b^", y, zp, "r-", y, zn, "r-") # causing travis build to fail

  return  acf_array
end
