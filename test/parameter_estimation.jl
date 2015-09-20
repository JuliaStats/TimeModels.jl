facts("Parameter Estimation") do

    context("Unspecified constraints default behaviour") do

        context("Without inputs") do
            mod1 = build_model()
            x, y = simulate(mod1, 100)
            p, fitmod = fit(y, build(randn(9)))
        end

        context("With inputs") do
            #test
        end

    end

    context("Parameter estimation with constraints") do

        context("Without inputs") do
            m, b, s, dt = 5, 2, 2, .1
            t = 0:dt:10
            y_true = m*t + b
            y_noisy = y_true + s*randn(length(t))
            lm, lm_params = zip(
                    (ParametrizedMatrix([1., 0, 0, 1], [0, 0, 1., 0]'', (2,2)), [1.]),
                    parametrize_none(0.00001*eye(2)),
                    parametrize_none([1., 0.]'), parametrize_diag(eye(1)),
                    parametrize_full([1., 1.]''), parametrize_none(0.001*eye(2)))
            fitm_params, fitm = fit(y_noisy, ParametrizedSSM(lm...), SSMParameters(lm_params...))
        end

        context("With inputs") do

            m, b, s, dt = 5, 2, 2, .1
            t = 0:dt:10
            y_true = m*t + b
            input = 100*[sin(t/2+0.1) sin(t/4+.1) cos(t/2+.1) cos(t/4+.1)] + 10
            y_noisy = [y_true zeros(length(t)) -y_true] +
                        100*[sin(t/2+.1)+sin(t/4+.1) sin(t/2)+cos(t/2+.1) cos(t/2+.1)+cos(t/4+.1)] + 10 + s*randn(length(t), 3)
            lm, lm_params = zip(
                  parametrize_none([1 0.1; 0 1]),
                  parametrize_none(zeros(2,4)),
                  parametrize_none(0.001*eye(2)),
                  parametrize_none([1. 0; 0 0; -1 0]),
                  parametrize_full(randn(3,4)),
                  parametrize_none(s^2*eye(3)),
                  parametrize_none([2., 5.]''), parametrize_none(0.001*eye(2)))
            fitm_params, fitm = fit(y_noisy, ParametrizedSSM(lm...), SSMParameters(lm_params...), u=input)

        end

    end

end

