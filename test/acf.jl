d  = readtime(Pkg.dir("TimeModels/test/data/unittest.csv"))
e  = simple_return(d["Close"])

acfd = acf(e, 10)

@assert 1.0                   == acfd[1]   
@assert 0.34071708386152755   == acfd[2]   # acf in R 0.34067593   
@assert 0.016977282568754647  == acfd[3]   # acf in R 0.01696415
@assert 0.041388411407948425  == acfd[4]   # acf in R 0.04132305
@assert 0.0533783066887273    == acfd[5]   # acf in R 0.05321190
@assert 0.036809522937119914  == acfd[6]   # acf in R 0.03669151
@assert -0.08783221241408778  == acfd[7]   # acf in R -0.08752100
@assert -0.026532083427867465 == acfd[8]   # acf in R -0.02639833  
@assert 0.10096389034532521   == acfd[9]   # acf in R 0.10043229
