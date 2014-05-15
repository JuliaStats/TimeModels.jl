# my_tests = ["Kalman.jl"]

# print_with_color(:cyan, "Running tests: ") 
# println("")

# for my_test in my_tests
#     print_with_color(:magenta, "**   ") 
#     print_with_color(:blue, "$my_test") 
#     println("")
#     include(my_test)
# end

module TimeModelTests

using FactCheck #, MarketData

tests = ["vanilla.jl"] #["Kalman.jl"]

for test in tests
  include(test)
end

exitstatus()

end
