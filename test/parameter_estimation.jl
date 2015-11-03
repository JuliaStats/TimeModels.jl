# Test model-fitting
function build(theta)
    F = diagm(theta[1])
    V = diagm(exp(theta[2]))
    G = reshape(theta[3:5], 3, 1)
    W = diagm(exp(theta[6:8]))
    x0 = [theta[9]]
    P0 = diagm(1e7)
    StateSpaceModel(F, V, G, W, x0, P0)
end

facts("Parameter Estimation") do

    #=
    context("Nonlinear solver") do
        mod1 = build_model()
        x, y = simulate(mod1, 100)
        fit(y, build, zeros(9))
    end
    =#

    #=
    context("Unspecified constraints default behaviour") do

        context("Without inputs") do
            mod1 = build_model()
            x, y = simulate(mod1, 100)
            #p, fitmod = fit(y, build(randn(9)))
        end

        context("With inputs") do
            #test
        end

    end
    =#

    context("Parameter estimation with constraints") do

        m, b, dt = 5, 2, .1
        t = 0:dt:100
        y_true = m*t + b

        context("Without inputs") do

            s = 2
            y_noisy = y_true + s*randn(length(t))

            lm = ParametrizedSSM( 
                    parametrize_none([1 0.1; 0 1])[1], #A
                    parametrize_none(ones(1,1))[1], #Q
                    parametrize_none([1. 0])[1], #C
                    parametrize_full(ones(1,1))[1], #R
                    parametrize_none([1., 1.]'')[1], #x1
                    parametrize_none(100*eye(2))[1], #P1
                    G = _->zeros(2,1) #G
                )

            lm_params = SSMParameters(1., R=rand(1))

            fitm_params, fitm = fit(y_noisy, lm, lm_params)
            @fact sqrt(fitm_params.R[1]) --> roughly(s, atol=0.1)

        end

        context("With inputs") do

            s1, s2, s3 = 1, 2, 3
            input = 100*[sin(t/2+0.1) sin(t/4+.1) cos(t/2+.1) cos(t/4+.1)] + 10
            y_noisy = [y_true zeros(length(t)) -y_true] +
                        [input[:,1]+input[:,2] input[:,1]+input[:,3] input[:,3]+input[:,4]] +
                        [s1 s2 s3] .* randn(length(t), 3)

            lm = ParametrizedSSM(
                  parametrize_none([1 0.1; 0 1])[1], #A
                  parametrize_none(eye(1))[1], #Q
                  parametrize_none([1. 0; 0 0; -1 0])[1], #C
                  parametrize_diag(ones(3))[1], #R
                  parametrize_none([2., 5.]'')[1], #x1
                  parametrize_none(0.001*eye(2))[1], #P1
                  B2=parametrize_none(zeros(2,4))[1], #B
                  G=_->zeros(2,1), #G
                  D2=parametrize_full(randn(3,4))[1] #D
            )
            lm_params = SSMParameters(1., R=rand(3), D=randn(12))
            fitm_params, fitm = fit(y_noisy, lm, lm_params, u=input)
            @fact sqrt(fitm_params.R) --> roughly([s1, s2, s3], atol=.1)

        end

    end

end

