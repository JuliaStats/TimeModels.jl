macro timemodels()
  println("")
  reload(Pkg.dir("TimeModels", "run_tests.jl"))
end
