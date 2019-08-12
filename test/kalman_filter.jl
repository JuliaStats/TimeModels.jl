#= using UnicodePlots =#

# Set up model
function build_model()
    F = diagm(0 => [1.0])
    V = diagm(0 => [2.0])
    G = reshape([1, 2, -0.5], 3, 1)
    W = diagm(0 => [8.0, 2.5, 4.0])
    x0 = randn(1)
    P0 = diagm(0 => [1e7])
    StateSpaceModel(F, V, G, W, x0, P0)
end

# Time varying model
function sinusoid_model(omega::Real; fs::Int=256, x0=[0.5, -0.5], W::AbstractFloat=0.1)
    F(n)  = [1.0 0; 0 1.0]
    V(n)  = diagm(0 => [1e-10, 1e-10])
    G(n)  = [cos(2*pi*omega*(1/fs)*n) -sin(2*pi*omega*(1/fs)*n)]
    w(n)  = diagm(0 => [W])
    P0    = diagm(0 => [1e-1, 1e-1])
    StateSpaceModel(F, V, G, w, x0, P0)
end


@testset "Kalman Filter" begin

    Random.seed!(1)

    @testset "Time invariant models" begin

        @testset "Building model" begin
            build_model()
        end

        @testset "Simulations" begin
            mod1 = build_model()
            x, y = TimeModels.simulate(mod1, 100)
        end

        @testset "Filtering" begin
            mod1 = build_model()
            x, y = TimeModels.simulate(mod1, 100)
            filt = kalman_filter(y, mod1)
        end

        @testset "Smoothing" begin
            mod1 = build_model()
            x, y = TimeModels.simulate(mod1, 100)

            smooth = kalman_filter(y, mod1) |> kalman_smooth
            smooth2 = kalman_smooth(y, mod1)

            @test smooth.smoothed ≈ smooth2.smoothed atol=1e-2
        end

        @testset "Missing data" begin
            mod1 = build_model()
            x, y = simulate(mod1, 100)
            y[1:9:end] .= NaN
            y[100] = NaN
            filt = kalman_filter(y, mod1)
            smooth = kalman_smooth(filt)
            smooth2 = kalman_smooth(y, mod1)
            @test !any(isnan, filt.filtered)
            @test !any(isnan, smooth.filtered)
            @test !any(isnan, smooth2.smoothed)
        end

        @testset "Return sizes" begin
            mod1 = build_model()
            x, y = TimeModels.simulate(mod1, 100)
            filt = kalman_filter(y, mod1)
            smooth = kalman_smooth(filt)
            smooth2 = kalman_smooth(y, mod1)

            @test size(filt.filtered) == size(filt.predicted)
            @test size(filt.filtered) == size(smooth.filtered)
            @test size(filt.filtered) == size(smooth.smoothed)
            @test size(filt.error_cov) == size(smooth.error_cov)
            @test size(filt.filtered) == size(smooth2.smoothed)
            @test size(filt.error_cov[:,:,1]) == size(smooth2.error_cov[1])
            @test size(filt.error_cov, 3) == length(smooth2.error_cov)
        end

        @testset "Linear regression test" begin
            m, b, s, dt = 5, 2, 2, .1
            t = collect(0:dt:10)
            y_true = m * t .+ b
            input = 100*[sin.(t./2) sin.(t./4) cos.(t./2) cos.(t./4)] .+ 10
            y_noisy = [y_true zeros(length(t)) -y_true] +
                        100*[sin.(t./2) + sin.(t./4) sin.(t./2) + cos.(t./2) cos.(t./2) + cos.(t./4)] .+ 10 .+ randn(length(t), 3)
            lm = StateSpaceModel([1 dt; 0 1], zeros(2,2),
                                  [1. 0; 0 0; -1 0], s*Matrix(1.0I, 3, 3),
                                  zeros(2), 100*Matrix(1.0I, 2, 2),
                                  B=zeros(2, 4), D=[1. 1 0 0; 1 0 1 0; 0 0 1 1])
            lm_filt = kalman_filter(y_noisy, lm, u=input)
            @test lm_filt.filtered[end,1] ≈ y_true[end, 1] atol=3*sqrt(lm_filt.error_cov[1,1,end])

            lm_smooth = kalman_smooth(lm_filt)
            stderr = sqrt.(lm_smooth.error_cov[1:1,1:1,:][:])
            @test lm_filt.filtered[end:end,:] == lm_smooth.smoothed[end:end,:]
            @test all(abs.(y_true - lm_smooth.smoothed[:,1]) .< 3*stderr)
            @test ones(size(t)) * lm_smooth.smoothed[1,2] ≈ lm_smooth.smoothed[:, 2] atol=1e-12

            # Repeat with DK smoother
            lm_smooth = kalman_smooth(y_noisy, lm, u=input)
            stderr = sqrt.(Float64[P[1,1] for P in lm_smooth.error_cov])
            @test all(abs.(y_true - lm_smooth.smoothed[:,1]) .< 3*stderr)
            @test ones(size(t)) * lm_smooth.smoothed[1,2] ≈ lm_smooth.smoothed[:, 2] atol=1e-12

        end

    end


    @testset "Time varying models" begin

        @testset "Building Model" begin
            sinusoid_model(40)
        end

        @testset "Simulations" begin
            Random.seed!(1)
            fs = 256
            mod2 = sinusoid_model(4, fs = 256)
            x, y = TimeModels.simulate(mod2, fs*2)

            #= display(lineplot(collect(1:size(x, 1)) / fs, vec(y), width = 120, title = "Original Data")) =#
        end

        @testset "Filtering" begin
            Random.seed!(1)
            fs = 256
            mod2 = sinusoid_model(4, fs = 256)
            x, y = TimeModels.simulate(mod2, fs*2)

            @testset "Correct initial guess" begin
                filt = kalman_filter(y, mod2)
                @test filt.predicted[end, :] ≈ [0.5, -0.5] atol=0.3
            end

            @testset "Incorrect initial guess" begin
                mod3 = sinusoid_model(4, fs = 256, x0=[1.7, -0.2])
                filt = kalman_filter(y, mod3)
                @test filt.predicted[end, :] ≈ [0.5, -0.5] atol= 0.3

            end

            @testset "Model error" begin
                mod4 = sinusoid_model(4, fs = 256, x0=[1.7, -0.2], W=3.0)
                filt = kalman_filter(y, mod4)
                @test filt.predicted[end, :] ≈ [0.5, -0.5] atol= 0.3
            end

            @testset "Standalone log-likelihood function works" begin
                loglik1 = kalman_filter(y, mod2).loglik
                loglik2 = loglikelihood(y, mod2)
                @test loglik1 ≈ loglik2
            end

        end

        @testset "Smoothing" begin
            Random.seed!(1)
            fs = 256
            mod2 = sinusoid_model(4, fs = 8192)
            x, y = TimeModels.simulate(mod2, fs*10)
            smooth = kalman_smooth(y, sinusoid_model(4, fs = 8192, x0=[1.7, -0.2]) )
            @test mean(smooth.smoothed; dims=1) ≈ [0.5 -0.5] atol= 0.1

            #= x_est = round(smooth.smoothed[end:end, :], 3) =#
            #= display(lineplot(collect(1:size(x, 1)) / fs, vec(smooth.smoothed[1:end, 1]), width = 120, title="Smoothed State 1: $(x_est[1])")) =#
            #= display(lineplot(collect(1:size(x, 1)) / fs, vec(smooth.smoothed[1:end, 2]), width = 120, title="Smoothed State 2: $(x_est[2])")) =#
        end

        @testset "Sparse regression test" begin
            # To be fair, not an exemplary example of sparse matrices in action
            # since the state covariance off-diagonals are small but nonzero
            n, m, s = 5000, 10, 5.
            x = 100*rand(n, m)
            coeffs = randn(m)
            y = x * coeffs + s*randn(n)

            I_m   = sparse(1.0I, m, m)
            I0_m  = spzeros(m,m)
            S     = diagm(0 => [s])
            mlm   = StateSpaceModel(
                _->I_m, # A
                _->I0_m, # V
                t->x[t:t,:], # C
                _->S, # W
                zeros(m), # x1
                sparse(1.0I, m, m) # P1
            )
            mlm_smooth = kalman_smooth(y, mlm)
            @test ones(n,m) .* mlm_smooth.smoothed[1:1,:] ≈ mlm_smooth.smoothed atol=1e-6
            @test vec(mlm_smooth.smoothed[1:1,:]) ≈ coeffs atol=0.1

            @testset "Correct failure" begin
                @test_throws MethodError StateSpaceModel([1 0.1; 0 1], zeros(2,3), zeros(2,2),
                    [1. 0; 0 0; -1 0], [1. 1 0 0; 1 0 1 0; 0 0 1 1], Matrix(1.0I, 3, 3), zeros(2), 100*Matrix(1.0I, 2, 2))
            end
        end

    end

end
