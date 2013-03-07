require("test.jl")

using TimeModels

my_tests = ["test/arima.jl",
            "test/garch.jl",
            "test/acf.jl"]

print_with_color(:cyan, "Running tests: ") 
println("")

for my_test in my_tests
    print_with_color(:magenta, "**   ") 
    print_with_color(:blue, "$my_test") 
    println("")
    include(my_test)
end
