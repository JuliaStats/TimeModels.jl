function em_checkmodel(pmodel::ParametrizedSSM, params::SSMParameters)

    allzero(x::Matrix) = all(x .== 0)

    m = pmodel(params)
    C = m.C(1)

    I_x, I_u  = Matrix(1.0I, m.nx, m.nx), Matrix(1.0I, m.nu, m.nu)
    I_q0      = diagm(0 => all(pmodel.G(1) .== 0, 2) |> vec |> float)
    I_r0      = diagm(0 => all(pmodel.H(1) .== 0, 2) |> vec |> float)
    M         = m.A(1) .!= 0 |> float
    #I_is_q    = M^m.nx  #TODO - work out indirect stochastic row selector

    @assert kron(I_x, I_r0 * C * I_q0) * pmodel.A2.D  |> allzero
    @assert kron(I_u, I_r0 * C * I_q0) * pmodel.B2.D  |> allzero
    @assert kron(I_x, I_r0) * pmodel.C2.D             |> allzero
    @assert kron(I_u, I_r0) * pmodel.D2.D             |> allzero
    @assert kron(I_x, I_q0) * pmodel.A2.D             |> allzero
    #@assert kron(I_u, I_is_q) * pmodel.B2.D           |> allzero

end #em_modelcheck
