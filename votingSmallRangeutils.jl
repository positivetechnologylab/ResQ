using Bloqade
using BloqadeSchema
using Yao
using Random:rand, MersenneTwister
using DifferentialEquations:solve
using Yao:projector, ghz_state, kron, ShiftGate, put, zero_state, product_state

using Plots

using LinearAlgebra: diagm
# const c6_coeff = get_device_capabilities().rydberg.c6_coefficient

# import Bloqade.YaoSubspaceArrayReg.space


function hatFunc(t, k, clocks)

    n = length(clocks)
    if k > 1 && clocks[k-1] <= t <= clocks[k]
        return (t - clocks[k-1]) / (clocks[k] - clocks[k-1])
    elseif k < n && clocks[k] <= t <= clocks[k+1]
        return (clocks[k+1]-t) / (clocks[k+1]-clocks[k])
    else
        return 0.0
    end
end

# function piecewisePulseNL(t, params, clocks, initVal)

#     level1 = params[1]*initVal + params[4]
#     level2 = params[2]*initVal + params[5]
#     level3 = params[3]*initVal + params[6]

#     vals = [0.0, level1, level1, level2, level2, level3, level3, 0.0]

#     # k = findlast(x->t>=x, clocks)
#     # println(t .>= clocks)
#     # println(clocks)
#     # println(k)

#     val = sum( map(k->(vals[k]*hatFunc(t, k, clocks)), 1:8))

#     return val
    
# end

function piecewisePulseNL(t, params, clocks, initVal)

    numIntervals = Int(length(params) / 2)
    levels = [params[i]*initVal + params[numIntervals+i] for j in [1,2], i in 1:numIntervals]

    vals = [0.0, levels... , 0.0]

    val = sum( map(k->(vals[k]*hatFunc(t, k, clocks)), 1:length(vals) ))

    return val
    
end


# function du_piecewisePulseNL(t, params, clocks, initVal)
#     return [ hatFunc(t, k, clocks)*b + hatFunc(t, k+1, clocks)*b for b in [initVal, 1] for k in 2:2:6 ]
# end

function du_piecewisePulseNL(t, params, clocks, initVal)
    return [ hatFunc(t, k, clocks)*b + hatFunc(t, k+1, clocks)*b for b in [initVal, 1] for k in 2:2:length(params) ]
end

#=

residual_computation performs the analog quantum res net computation, and returns the gradients w.r.t the parameters

INPUTS:
    features : a vector of the features used as input to the model. Formatted as [hinput, Ω0, Δg0, ϕ0, Δl0]. θx and θz ∈ ℜᴺ , all others ∈ ℜ.
    parameters : a vector of the parameters. Formatted as [pΩ, pΔg, pϕ, pΔl, hparam].
    hyperparams : a vector of the hyperparameters
    observable : the measured quantum operator used for output. Should be compatible with Yao.expect

OUTPUTS:
    value : measured value of the observable for the given features and parameters
    grad : vector of gradients with respect to each parameter
=#
function residual_computation(hinput, Ω0, Δg0, ϕ0, Δl0, pΩ, pΔg, pϕ, pΔl, hparam, bint, latticeType, numAtoms, latticeConstant, numTiers, observable; computeGrad=true)
    
    E_M = r->Yao.expect(observable(numAtoms), r)

    minDt = 0.05
    # initializationTime = 12 * minDt
    computeTime = numTiers * (4 * minDt) + minDt
    clocks = vec([4*i + j for j in [0,1], i in 0:numTiers]) * minDt

    taus =  sort!( rand(MersenneTwister(1), Float64, bint))*computeTime #+ initializationTime # generate integration sampling times for grad calc

    # construct lattices
    if latticeType == "square"
        lattice = generate_sites(SquareLattice(), Int(sqrt(numAtoms)), Int(sqrt(numAtoms)) ; scale=latticeConstant)
    elseif latticeType == "triangle"
        lattice = generate_sites(TriangularLattice(), Int(sqrt(numAtoms)), Int(sqrt(numAtoms)); scale=latticeConstant)
    elseif latticeType == "chain"
        lattice = generate_sites(ChainLattice(), numAtoms; scale=latticeConstant)
    else
        error("Invalid lattice type, specify one of \"square\", \"triangle\", or \"chain\"")
    end
    
    # println(lattice)
    # display(Bloqade.plot(lattice))
    # computation pulses and grid sites
    compute_reg = zero_state(numAtoms)

    hgrid = [hinput[1], hparam[1], hinput[2], hparam[2]]

    Ωpulse =  t -> piecewisePulseNL(t, pΩ, clocks, Ω0)
    # Δgpulse =  t -> piecewisePulseNL(t, pΔg, clocks, Δg0)
    Δpulse = [t -> piecewisePulseNL(t, pΔg, clocks, Δg0) + piecewisePulseNL(t, pΔl, clocks, Δl0) * h for h in hgrid]
    ϕpulse = BloqadeWaveforms.constant(duration=clocks[end], value=-pi/2) # -pi/2 # need to change Xphase term in grad calc if we want non constant phase

    # p = Plots.plot(clocks, map(t->Ωpulse(t), clocks))
    # p = Plots.plot(clocks, map(t->piecewisePulseNL(t, pΔg, clocks, Δg0), clocks))
    # p = Plots.plot(clocks, map(t->-pi/2, clocks))
    # p = Plots.plot(clocks, map(t->piecewisePulseNL(t, pΔl, clocks, Δl0), clocks))

    # # Plots.ylims!(p, (-1.6, -1.5))
    # # Plots.yticks!(p, 0:0.5:6.0)
    # display(p)

    compute_hamiltonian = rydberg_h(lattice, Ω=Ωpulse, Δ=Δpulse, ϕ=ϕpulse)
    compute_diffeq = SchrodingerProblem(compute_reg, computeTime, compute_hamiltonian, save_on=true, saveat=taus)

    compute_result = solve(compute_diffeq, alg=DP8())

    numΔterms = numAtoms
    numΩterms = numAtoms
    numTotalTerms = numΔterms + numΩterms

    p_minus = zeros(numΔterms)
    p_plus = zeros(numΔterms)

    f_rabi = zeros(length(pΩ), bint)
    f_detune = zeros(length(pΔg) + length(pΔl) + length(hparam), bint)

    localDetuneGradGrid = [iseven(l) && m==l/2 ? 1.0 : 0.0 for m in 1:length(hparam), l in 1:numAtoms]

    # discrete_hamiltonian = hardware_transform(compute_hamiltonian)
    # hardware_schema = to_schema(discrete_hamiltonian)
    # h_braket = to_braket_ahs_ir(h_schema)

    # print(h_braket)

    # if computeGrad 
    #     println(compute_result.u[1][1:3])
    #     println(compute_result.t)
    #     # println(taus)
    # end
    
    if computeGrad

        for (k, intm_state, tau) in zip(1:bint, compute_result.u, compute_result.t)

            # do various perturbations on copies of this state
            # batch_amplitudes = zeros(Int(2^numAtoms), numTotalTerms)

            for j in 1:numΩterms, s in [-1,+1]

                copy_reg = arrayreg(copy(intm_state))

                # if k == 1 && j == 1 && s == -1
                #     println("pre_copy : ", intm_state[1:3])
                #     println("copy reg: ", copy_reg.state[1:3])
                # end

                perturb_time = (1 + 3/4*s)*pi
                h_j = chain(numAtoms, put(numAtoms, j=>XPhase(-pi/2)))

                apply!(copy_reg, rot(h_j, perturb_time))

                grad_problem = SchrodingerProblem(copy_reg, (tau, computeTime), compute_hamiltonian)
                emulate!(grad_problem)

                measure_result = E_M(copy_reg)

                if s == 1
                    p_plus[j] = measure_result #(1 / b_obs) * sum( x ) 
                    # remove these extra operations, since doing expectation value so b_obs=1 always
                    # could instead do Yao.measure! to see the effects of shot noise
                elseif  s == -1
                    p_minus[j] = measure_result #(1 / b_obs) * sum( x )
                end
            end

            deriv_u = 1/2 * du_piecewisePulseNL(tau, pΩ, clocks, Ω0) * ones(1, numΩterms)
            f_rabi[:,k] = deriv_u*(p_minus - p_plus) #/ sqrt(2)
            
            # if k == 1
            #     println("intm state", intm_state[1:3])
            #     println("tau", tau)
            #     # println("p_plus", p_plus)
            #     println("")
            # end

            for j in 1:numΔterms, s in [-1, +1]

                copy_reg = arrayreg(copy(intm_state))

                perturb_time = (1 + 3/4*s)*pi

                perturb = ShiftGate(perturb_time)
                apply!(copy_reg, put(numAtoms, j=>perturb) )

                grad_problem = SchrodingerProblem(copy_reg, (tau, computeTime), compute_hamiltonian)
                emulate!(grad_problem)

                measure_result = E_M(copy_reg)

                if s == 1
                    p_plus[j] = measure_result #(1 / b_obs) * sum( x ) 
                elseif  s == -1
                    p_minus[j] = measure_result #(1 / b_obs) * sum( x )
                end

            end

            deriv_u = reduce(vcat, [du_piecewisePulseNL(tau, pΔg, clocks, Δg0) * ones(1, numΔterms) , du_piecewisePulseNL(tau, pΔl, clocks, Δl0) * hgrid',  piecewisePulseNL(tau, pΔl, clocks, Δl0) * localDetuneGradGrid])
            # deriv_u = du_piecewisePulseNL(tau, pΔg, clocks, Δg0) * ones(1, numΔterms)
            f_detune[:,k] = deriv_u*(p_minus - p_plus) / sqrt(2)
            
            # make batched register with all different perturbation terms
            # perturb_batch = arrayreg(batch_amplitudes, nbatch=numTotalTerms)

        end

        grad = reduce(vcat, ( (computeTime / bint) * sum(f_rabi, dims=(2)) , (computeTime / bint) * sum(f_detune, dims=(2))))
    else
        grad = nothing
    end

    val = E_M(compute_reg)
    

    return val, grad
end

function trainQNet(lossFunc, init_guess, batched_data, batched_labels, numEpochs, numStepsPerBatch,  solver)

    current_guess = init_guess
    numBatches = length(batched_labels)

    losses = ones(numBatches, numEpochs)

    # solver = AdaMax(alpha=0.002)

    for epoch in 1:numEpochs
        println("Epoch: $epoch of $numEpochs")
        batch_num = 1
        for (b, l) in zip(batched_data, batched_labels)
            println("batch num: $batch_num")
            
            # define closure that covers the training data, for the optimizer
            fg!(F, G, x) = lossFunc(F, G, x, b, l)
            sol = Optim.optimize(Optim.only_fg!(fg!), current_guess, solver, Optim.Options(show_trace=true, store_trace = false, show_warnings = true, iterations=numStepsPerBatch))

            current_guess = sol.minimizer
            losses[batch_num, epoch] = sol.minimum

            # println(sol.minimum)
            # println(sol.minimizer)
            batch_num += 1
        end
    end

    return current_guess, losses
end

function inferQNet(data, params)
    numAtoms = 4
    numSamples = size(data)[1]

    # this should be made more modular
    pΩ = params[1:NUM_TIERS*2]
    pΔg = params[NUM_TIERS*2+1:4*NUM_TIERS]
    pϕ = 0
    pΔl = params[4*NUM_TIERS+1:6*NUM_TIERS]
    hparam = params[6*NUM_TIERS:6*NUM_TIERS+div(numAtoms, 2)]

    bint = 20
    # latticeType = "chain"
    
    # latticeConstant = 10
    numTiers = NUM_TIERS
    # observable = observable = numq -> put(numq, 1=>projector(product_state(bit"1")))
    if measurement_scheme == "all"
        observable = numq -> 1/4 * (put(numq, 1=>projector(product_state(bit"1")) ) + put(numq, 2=>projector(product_state(bit"1"))) + put(numq, 3=>projector(product_state(bit"1")) ) + put(numq, 4=>projector(product_state(bit"1")) )  )
    elseif measurement_scheme == "half"
        observable = numq -> 1/2 * (put(numq, 1=>projector(product_state(bit"1")) ) + put(numq, 3=>projector(product_state(bit"1")) ) )
    else
        error("incorrect observable selection. Do \"all\" or \"half\"")
    end

    res_logits = zeros(numSamples)

    for idx in 1:numSamples

        println(idx, "/$(numSamples)")
        hinput = data[idx, 3:4]
        Ω0 = data[idx, 2]
        Δg0 = data[idx, 1]
        ϕ0 = 0
        # Δl0 = -data[idx,4]
        Δl0 = data[idx, 5]

        q_result, _ = residual_computation(hinput, Ω0, Δg0, ϕ0, Δl0, pΩ, pΔg, pϕ, pΔl, hparam, bint, LATTICE_TYPE, numAtoms, LATTICE_CONSTANT, numTiers, observable; computeGrad=false)
        
        res_logits[idx] = q_result
    end

    hard_labels = round.(res_logits)

    return hard_labels, res_logits
end


# singleOb = numQ -> put(numQ, 1=>projector(product_state(bit"1")))

# @time asdf, asdfasdf = residual_computation([1, 1], 4.0, 4.0 , 0, 0, [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 1], 20, "square", 4, 10.0, 3, singleOb)

# asdf.u

# b = arrayreg(reduce(hcat, asdf.u), nbatch=20)

# c = clone(arrayreg(asdf.u[1]), 2)

# for a in 1:5
#     for b in 1:5
#         print( ( (-1)^a * (-1)^b )==1 ? "o" : "x" )
#     end 
#     println("")
# end

# a = [0.5+0im, 0.5, 0.5, 0.5]
# reg = arrayreg(a)

# taus = [0.3, 0.5]
# # reg = zero_state(2)
# positions = [[0,0], [0, 10]]
# rh = rydberg_h(positions; Ω=5.0, Δ=-10.0, ϕ=-pi/2)
# prob = SchrodingerProblem(reg, 1.0, rh, save_on=true, saveat=taus)
# res = Bloqade.solve(prob, alg=DP8())
# println(reg.state)

# for (state, t) in zip(res.u, res.t)
#     println("time: ", t)
#     println("state: ", state)
# end
# res.u
# res.t

# new_reg = zero_state(2)
# new_prob = SchrodingerProblem(new_reg, 0.5, rh)
# emulate!(new_prob)
# println(new_reg.state)

# println(res.u[2])
# inputMask = vec([isodd(i+j) for i in 1:4, j in 1:4])
# inputMask = vec([isodd(i+j) for i in 1:4, j in 1:4])

# paramMask =  [!b for b in inputMask]

# asdfasdf = rand(Float16, (4, 4))

# v = @view asdfasdf[inputMask]

# v .= 0

# print(asdfasdf)


