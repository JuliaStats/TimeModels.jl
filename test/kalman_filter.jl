#= using UnicodePlots =#

# Set up model
function build_model()
    F = diagm([1.0])
    V = diagm([2.0])
    G = reshape([1, 2, -0.5], 3, 1)
    W = diagm([8.0, 2.5, 4.0])
    x0 = randn(1)
    P0 = diagm([1e7])
    mod1 = StateSpaceModel(F, V, G, W, x0, P0)
end

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

# Time varying model
function sinusoid_model(omega::Real; fs::Int=256, x0=[0.5, -0.5], W::FloatingPoint=0.1)
    F  = [1.0 0; 0 1.0]
    V  = diagm([1e-10, 1e-10])
    function G1(n);  cos(2*pi*omega*(1/fs)*n); end
    function G2(n); -sin(2*pi*omega*(1/fs)*n); end
    G = reshape([G1, G2], 1, 2)
    W = diagm([W])
    P0 = diagm([1e-1, 1e-1])
    StateSpaceModel(F, V, G, W, x0, P0)
end


facts("Kalman Filter") do

    srand(1)

    context("Time invariant models") do

        context("Building model") do
            build_model()
        end

        context("Simulations") do
            mod1 = build_model()
            x, y = TimeModels.simulate(mod1, 100)
        end

        context("Filtering") do
            mod1 = build_model()
            x, y = TimeModels.simulate(mod1, 100)
            filt = kalman_filter(y, mod1)
        end

        context("Smoothing") do
            mod1 = build_model()
            x, y = TimeModels.simulate(mod1, 100)
            smooth = kalman_smooth(y, mod1)
        end

        context("Model fitting") do
            mod1 = build_model()
            x, y = simulate(mod1, 100)
            fit(y, build, zeros(9))
        end

        context("Missing data") do
            mod1 = build_model()
            x, y = simulate(mod1, 100)
            y[1:9:end] = NaN
            y[100] = NaN
            filt = kalman_filter(y, mod1)
            smooth = kalman_smooth(y, mod1)
            @fact any(isnan(filt.filtered)) --> false
            @fact any(isnan(smooth.filtered)) --> false
        end

        context("Return sizes") do
            mod1 = build_model()
            x, y = TimeModels.simulate(mod1, 100)
            filt = kalman_filter(y, mod1)
            smooth = kalman_smooth(y, mod1)

            @fact size(filt.filtered) --> size(filt.predicted)
            @fact size(filt.filtered) --> size(smooth.filtered)
            @fact size(filt.filtered) --> size(smooth.smoothed)
            @fact size(filt.error_cov) --> size(smooth.error_cov)
        end
    end


    context("Time varying models") do

        context("Building Model") do
            sinusoid_model(40)
        end

        context("Simulations") do
            srand(1)
            fs = 256
            mod2 = sinusoid_model(4, fs = 256)
            x, y = TimeModels.simulate(mod2, fs*2)

            #= display(lineplot(collect(1:size(x, 1)) / fs, vec(y), width = 120, title = "Original Data")) =#
        end

        context("Filtering") do
            srand(1)
            fs = 256
            mod2 = sinusoid_model(4, fs = 256)
            x, y = TimeModels.simulate(mod2, fs*2)

            context("Correct initial guess") do
                filt = kalman_filter(y, mod2)
                @fact filt.predicted[end, :] --> roughly([0.5 -0.5]; atol= 0.3)

                #= x_est = round(filt.predicted[end, :], 3) =#
                #= display(lineplot(collect(1:size(x, 1)) / fs, vec(filt.predicted[:, 1]), width = 120, title="Filtered State 1: $(x_est[1])")) =#
                #= display(lineplot(collect(1:size(x, 1)) / fs, vec(filt.predicted[:, 2]), width = 120, title="Filtered State 2: $(x_est[2])")) =#
            end

            context("Incorrect initial guess") do
                mod3 = sinusoid_model(4, fs = 256, x0=[1.7, -0.2])
                filt = kalman_filter(y, mod3)
                @fact filt.predicted[end, :] --> roughly([0.5 -0.5]; atol= 0.3)

            end

            context("Model error") do
                mod4 = sinusoid_model(4, fs = 256, x0=[1.7, -0.2], W=3.0)
                filt = kalman_filter(y, mod4)
                @fact filt.predicted[end, :] --> roughly([0.5 -0.5]; atol= 0.3)
            end

        end

        context("Smoothing") do
            srand(1)
            fs = 256
            mod2 = sinusoid_model(4, fs = 8192)
            x, y = TimeModels.simulate(mod2, fs*10)
            smooth = kalman_smooth(y, sinusoid_model(4, fs = 8192, x0=[1.7, -0.2]) )
            @fact mean(smooth.smoothed, 1) --> roughly([0.5 -0.5]; atol= 0.1)

            #= x_est = round(smooth.smoothed[end, :], 3) =#
            #= display(lineplot(collect(1:size(x, 1)) / fs, vec(smooth.smoothed[1:end, 1]), width = 120, title="Smoothed State 1: $(x_est[1])")) =#
            #= display(lineplot(collect(1:size(x, 1)) / fs, vec(smooth.smoothed[1:end, 2]), width = 120, title="Smoothed State 2: $(x_est[2])")) =#
        end


    end

    context("Linear regression test") do
      m, b, s, dt = 5, 2, 2, .1
      t = 0:dt:10
      y_true = m*t + b
      y_noisy = y_true + s*randn(length(t))
      lm = StateSpaceModel([1 dt; 0 1], zeros(2,2), [1. 0], s*eye(1), zeros(2), 100*eye(2))
      lm_filt = kalman_filter(y_noisy, lm)
      @fact lm_filt.filtered[end,1] --> roughly(y_true[end], atol=4*sqrt(lm_filt.error_cov[1,1,end]))
      lm_smooth = kalman_smooth(y_noisy, lm)
      @fact all((y_true - lm_smooth.smoothed[:,1]) .< 4*sqrt(lm_smooth.error_cov[1,1,:][:])) --> true
    end


    context("Correct failure") do
    end
end


