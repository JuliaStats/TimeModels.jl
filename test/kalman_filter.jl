#= using UnicodePlots =#

# Set up model
function build_model()
    F = diagm([1.0])
    V = diagm([2.0])
    G = reshape([1, 2, -0.5], 3, 1)
    W = diagm([8.0, 2.5, 4.0])
    x0 = randn(1)
    P0 = diagm([1e7])
    StateSpaceModel(F, V, G, W, x0, P0)
end

# Time varying model
function sinusoid_model(omega::Real; fs::Int=256, x0=[0.5, -0.5], W::AbstractFloat=0.1)
    F(n)  = [1.0 0; 0 1.0]
    V(n)  = diagm([1e-10, 1e-10])
    G(n)  = [cos(2*pi*omega*(1/fs)*n) -sin(2*pi*omega*(1/fs)*n)]
    w(n)  = diagm([W])
    P0    = diagm([1e-1, 1e-1])
    StateSpaceModel(F, V, G, w, x0, P0)
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

            smooth = kalman_filter(y, mod1) |> kalman_smooth
            smooth2 = kalman_smooth(y, mod1)

            @fact smooth.smoothed --> roughly(smooth2.smoothed, atol=1e-2)
        end

        context("Missing data") do
            mod1 = build_model()
            x, y = simulate(mod1, 100)
            y[1:9:end] = NaN
            y[100] = NaN
            filt = kalman_filter(y, mod1)
            smooth = kalman_smooth(filt)
            smooth2 = kalman_smooth(y, mod1)
            @fact any(isnan(filt.filtered)) --> false
            @fact any(isnan(smooth.filtered)) --> false
            @fact any(isnan(smooth2.smoothed)) --> false
        end

        context("Return sizes") do
            mod1 = build_model()
            x, y = TimeModels.simulate(mod1, 100)
            filt = kalman_filter(y, mod1)
            smooth = kalman_smooth(filt)
            smooth2 = kalman_smooth(y, mod1)

            @fact size(filt.filtered) --> size(filt.predicted)
            @fact size(filt.filtered) --> size(smooth.filtered)
            @fact size(filt.filtered) --> size(smooth.smoothed)
            @fact size(filt.error_cov) --> size(smooth.error_cov)
            @fact size(filt.filtered) --> size(smooth2.smoothed)
            @fact size(filt.error_cov) --> size(smooth2.error_cov)
        end

        context("Linear regression test") do
            m, b, s, dt = 5, 2, 2, .1
            t = 0:dt:10
            y_true = m*t + b
            input = 100*[sin(t/2) sin(t/4) cos(t/2) cos(t/4)] + 10
            y_noisy = [y_true zeros(length(t)) -y_true] +
                        100*[sin(t/2)+sin(t/4) sin(t/2)+cos(t/2) cos(t/2)+cos(t/4)] + 10 + randn(length(t), 3)
            lm = StateSpaceModel([1 dt; 0 1], zeros(2,2),
                                  [1. 0; 0 0; -1 0], s*eye(3),
                                  zeros(2), 100*eye(2),
                                  B=zeros(2, 4), D=[1. 1 0 0; 1 0 1 0; 0 0 1 1])
            lm_filt = kalman_filter(y_noisy, lm, u=input)
            @fact lm_filt.filtered[end,1] --> roughly(y_true[end, 1], atol=3*sqrt(lm_filt.error_cov[1,1,end]))

            lm_smooth = kalman_smooth(lm_filt)
            stderr = sqrt(lm_smooth.error_cov[1,1,:][:])
            @fact lm_filt.filtered[end,:] --> lm_smooth.smoothed[end,:]
            @fact all(abs(y_true - lm_smooth.smoothed[:,1]) .< 3*stderr) --> true
            @fact ones(t) * lm_smooth.smoothed[1,2] --> roughly(lm_smooth.smoothed[:, 2], atol=1e-12)

            # Repeat with DK smoother
            lm_smooth = kalman_smooth(y_noisy, lm, u=input)
            stderr = sqrt(lm_smooth.error_cov[1,1,:][:])
            @fact all(abs(y_true - lm_smooth.smoothed[:,1]) .< 3*stderr) --> true
            @fact ones(t) * lm_smooth.smoothed[1,2] --> roughly(lm_smooth.smoothed[:, 2], atol=1e-12)

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

    context("Correct failure") do
        @fact_throws StateSpaceModel([1 0.1; 0 1], zeros(2,3), zeros(2,2),
                [1. 0; 0 0; -1 0], [1. 1 0 0; 1 0 1 0; 0 0 1 1], s*eye(3), zeros(2), 100*eye(2))
    end

end

