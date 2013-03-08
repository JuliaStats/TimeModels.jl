d  = read_csv_for_testing(Pkg.dir("TimeSeries", "test", "data"), "spx.csv")
e  = simple_return(d["Close"])

acfd = acf(e, 10)

@assert 1.0                  == acfd[1]   
@assert 0.3407170838615277   == acfd[2]   # acf in R 0.34067593   
@assert 0.016977282568754616 == acfd[3]   # acf in R 0.01696415
@assert 0.04138841140794836  == acfd[4]   # acf in R 0.04132305
@assert 0.053378306688727326 == acfd[5]   # acf in R 0.05321190
@assert 0.03680952293711989  == acfd[6]   # acf in R 0.03669151
@assert -0.08783221241408777 == acfd[7]   # acf in R -0.08752100
@assert -0.0265320834278675  == acfd[8]   # acf in R -0.02639833  
@assert 0.10096389034532514  == acfd[9]   # acf in R 0.10043229


