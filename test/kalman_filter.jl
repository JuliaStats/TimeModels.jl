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


facts("Kalman Filter") do

    srand(1)

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
        @fact filt.loglik --> 17278.89579046732
    end

    context("Smoothing") do
        mod1 = build_model()
        x, y = TimeModels.simulate(mod1, 100)
        smooth = kalman_smooth(y, mod1)
    end

    context("Model fitting") do
        # Why are two simulates required? is it a bug?
        srand(1)
        mod1 = build_model()
        x, y = simulate(mod1, 100)
        x, y = simulate(mod1, 100)
        theta0 = zeros(9)
        fit(y, build, theta0)
    end

    context("Correct failure") do
    end
end


