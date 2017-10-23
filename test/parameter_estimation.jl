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

@testset "Parameter Estimation" begin

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

    @testset "Parameter estimation with constraints" begin

        m, b, dt = 5, 2, .1
        t = 0:dt:200
        y_true = m*t + b

        @testset "Without exogenous inputs" begin

            s = 2
            y_noisy = y_true + s*randn(length(t))

            lm = ParametrizedSSM(
                    parametrize_none([1 0.1; 0 1]), #A
                    parametrize_none(ones(1,1)), #Q
                    parametrize_none([1. 0]), #C
                    parametrize_diag([1.]), #R
                    parametrize_none([1. 1.]'), #x1
                    parametrize_none(100*eye(2)), #P1
                    G = _->zeros(2,1) #G
                )
            lm_params = SSMParameters(1., R=rand(1))
            fitm_params, fitm = fit(y_noisy, lm, lm_params)
            @test sqrt(fitm_params.R[1]) ≈ s atol=0.1

            A = [.5 .1 .4; .25 .8 .5; .25 .1 .1]
            model = StateSpaceModel(A, diagm([.01,.01,.01]),
                                        eye(3), zeros(3,3), ones(3)/3, 0eye(3))
            _, y = simulate(model, 100)
            lm = ParametrizedSSM(
                    ParametrizedMatrix([0, .25, .75, .1, 0, .9, 0, .5, .5], [
                        1. 0 0; 0 0 0; -1 0 0; 0 0 0; 0 1 0; 0 -1 0; 0 0 1; 0 0 0; 0 0 -1
                    ], (3,3)), #A
                    parametrize_diag(ones(3)), #Q
                    parametrize_none(eye(3)), #C
                    parametrize_none(ones(1,1)), #R
                    parametrize_none(ones(3,1)/3), #x1
                    parametrize_none(100*eye(3)), #P1
                    H = _->zeros(3,1) #H
                )

            lm_params = SSMParameters(1., A=rand(3), Q=rand(3))
            fitm_params, fitm = fit(y, lm, lm_params)
            @test fitm_params.A ≈ [.5, .8, .4] atol=0.2
            @test fitm_params.Q ≈ [.01, .01, .01] atol=0.05

            coeffs = randn(3)
            X = randn(200,3)
            Y = X * coeffs + .4*randn(200)
            lm = ParametrizedSSM(
                parametrize_none(eye(3)), #A
                parametrize_none(eye(1)), #Q
                parametrize_none(eye(3)), #C2
                parametrize_diag(ones(1)), #R
                parametrize_full(zeros(3,1)), #x1
                parametrize_none(eye(3)), #S
                G=_->zeros(3,1), C1=t->X[t:t, :]
            )

            lm_params = SSMParameters(1., R=ones(1), x1=100*randn(3))
            fitm_params, fitm = fit(Y, lm, lm_params)
            @test fitm_params.x1   ≈ coeffs atol=0.1
            @test fitm_params.R[1] ≈ .16    atol=0.05

            lm = ParametrizedSSM(
                parametrize_none(eye(3)), #A
                parametrize_none(eye(1)), #Q
                parametrize_none(eye(3)), #C2
                parametrize_none(.16*ones(1,1)), #R
                parametrize_none(coeffs+[.2,.3,.5].*randn(3,1)), #x1
                parametrize_diag(ones(3)), #S
                G=_->zeros(3,1), C1=t->X[t:t, :]
            )
            lm_params = SSMParameters(1., S=100*ones(3))
            fitm_params, fitm = fit(Y, lm, lm_params)
            @test fitm_params.S ≈ [.2,.3,.5].^2 atol=0.15

        end

        @testset "With exogenous inputs" begin

            s1, s2, s3 = 1., 2., 3.
            input = 100.*[sin.(t./2 .+ 0.1) sin.(t./4 .+ .1) cos.(t./2 .+ .1) cos.(t./4 .+ .1)] .+ 10
            y_noisy = [y_true zeros(length(t)) -y_true] +
                        [input[:,1] + input[:,2] input[:,1] + input[:,3] input[:,3] + input[:,4]] +
                        [s1 s2 s3] .* randn(length(t), 3)

            lm = ParametrizedSSM(
                  parametrize_none([1 0.1; 0 1]), #A
                  parametrize_none(eye(1)), #Q
                  parametrize_none([1. 0; 0 0; -1 0]), #C
                  parametrize_diag(ones(3)), #R
                  parametrize_none([2. 5.]'), #x1
                  parametrize_none(0.001*eye(2)), #P1
                  B2=parametrize_none(zeros(2,4)), #B
                  G=_->zeros(2,1), #G
                  D2=parametrize_full(randn(3,4)) #D
            )
            lm_params = SSMParameters(1., R=rand(3), D=randn(12))
            fitm_params, fitm = fit(y_noisy, lm, lm_params, u=input)
            @test fitm_params.D       ≈ vec([1. 1 0 0; 1 0 1 0; 0 0 1 1]) atol=.1
            @test sqrt.(fitm_params.R) ≈ [s1, s2, s3] atol=.15

            y_noisy = [0 0 0;
                  [input[:,1]+input[:,2] input[:,1]+input[:,3] input[:,3]+input[:,4]][1:end-1, :]] +
                  [s1 s2 s3] .* randn(length(t), 3)

            lm = ParametrizedSSM(
                  parametrize_none(zeros(3,3)), #A
                  parametrize_diag(ones(3)), #Q
                  parametrize_none(eye(3)), #C
                  parametrize_none(eye(1)), #R
                  parametrize_none(zeros(3,1)), #x1
                  parametrize_none(0.001*eye(3)), #P1
                  B2=parametrize_none([1. 1 0 0; 1 0 1 0; 0 0 1 1]), #B
                  H=_->zeros(3,1), #H
                  D2=parametrize_none(zeros(3,4)), #D
            )
            lm_params = SSMParameters(1., Q=rand(3))
            fitm_params, fitm = fit(y_noisy, lm, lm_params, u=input)
            @test sqrt.(fitm_params.Q) ≈ [s1, s2, s3] atol=.1

            lm = ParametrizedSSM(
                  parametrize_none(zeros(3,3)), #A
                  parametrize_none(diagm([s1, s2, s3])), #Q
                  parametrize_none(eye(3)), #C
                  parametrize_none(eye(1)), #R
                  parametrize_none(zeros(3,1)), #x1
                  parametrize_none(0.001*eye(3)), #P1
                  B2=parametrize_full(randn(3,4)), #B
                  H=_->zeros(3,1), #H
                  D2=parametrize_none(zeros(3,4)), #D
            )
            lm_params = SSMParameters(1., B=randn(12))
            fitm_params, fitm = fit(y_noisy, lm, lm_params, u=input)
            @test fitm_params.B ≈ vec([1. 1 0 0; 1 0 1 0; 0 0 1 1]) atol=.11

            y_noisy = [0; .5input[1:end-1,1] - .8input[1:end-1,2] - .3input[1:end-1,3] + .7input[1:end-1,4]] + s2*randn(length(t))

            lm = ParametrizedSSM(
                  parametrize_none(zeros(4,4)), #A
                  parametrize_none(eye(1)), #Q
                  parametrize_full(ones(1,4)), #C
                  parametrize_full(diagm(s2)), #R
                  parametrize_none(zeros(4,1)), #x1
                  parametrize_none(0.001*eye(4)), #P1
                  B2=parametrize_none(eye(4)), #B
                  G=_->zeros(4,1), #H
                  D2=parametrize_none(zeros(1,4)), #D
            )
            lm_params = SSMParameters(1., C=randn(4), R=rand(1))
            fitm_params, fitm = fit(y_noisy, lm, lm_params, u=input)
            @test fitm_params.C ≈ [.5, -.8, -.3, .7] atol=.1
            @test sqrt(fitm_params.R[1]) ≈ s2 atol=.1

        end

    end

end

