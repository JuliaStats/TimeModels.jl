#  #plot with Winston

##  y = linspace(0,n-1,n)
## plots acf function in the style of R but with color and without bars for values
##  x = acf_array
##  zp = 0.05(ones(n))
##  zn = -0.05(ones(n))
##
##  plot(y, x, "b^", y, zp, "r-", y, zn, "r-") # causing travis build to fail
