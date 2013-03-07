acf = function(x::Array, n::Int)

  acf_array = ones(n)
 
  for i in 2:n
    acf_array[i] = cor(x[i+1:end], lag(x, i))
  end
  acf_array
end



 
