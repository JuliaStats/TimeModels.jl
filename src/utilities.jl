function em_checkmodel(pmodel::ParametrizedSSM, params::SSMParameters)
  
    allzero(x::Matrix) = all(x .== 0)

    m = pmodel(params)
    C = m.C(1)

    I_x, I_u  = eye(m.nx), eye(m.nu)
    I_q0      = all(pmodel.G(1) .== 0, 2) |> vec |> float |> diagm
    I_r0      = all(pmodel.H(1) .== 0, 2) |> vec |> float |> diagm
    M         = m.A(1) .!= 0 |> float
    #I_is_q    = M^m.nx  #TODO - work out indirect stochastic row selector

    @assert kron(I_x, I_r0 * C * I_q0) * pmodel.A2.D  |> allzero
    @assert kron(I_u, I_r0 * C * I_q0) * pmodel.B2.D  |> allzero
    @assert kron(I_x, I_r0) * pmodel.C2.D             |> allzero
    @assert kron(I_u, I_r0) * pmodel.D2.D             |> allzero
    @assert kron(I_x, I_q0) * pmodel.A2.D             |> allzero
    #@assert kron(I_u, I_is_q) * pmodel.B2.D           |> allzero

end #em_modelcheck
