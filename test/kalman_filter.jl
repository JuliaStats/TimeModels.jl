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
            theta0 = zeros(9)
            fit(y, build, theta0)
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

    end

    context("Correct failure") do
    end
end


