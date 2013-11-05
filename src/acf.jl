acf = function(x::Array, n::Int)

  acf_array = ones(n)
  [acf_array[i] = cor(x[1:end-(i-1)], x[i:end]) for i in 2:n]
  
  #plot with Winston
  y = linspace(0,n-1,n)
  x = acf_array
  plot(y, x, "b^", y, x, "g:")

  return  acf_array
end

# cor(e[1:end-1    ], e[2:end   ]) # n = 2
# cor(e[1:end-2    ], e[3:end  ]) # n = 3
