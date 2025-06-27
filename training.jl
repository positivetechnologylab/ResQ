const datasettype = ARGS[1]
const LATTICE_TYPE= ARGS[2]
const LATTICE_CONSTANT::Float64 = parse(Float64, ARGS[3])
const TRAIN_DATA_PERCENTAGE::Float64 = 80#parse(Float64, ARGS[4])
const measurement_scheme = "all"
const NUM_TIERS::Int64 = 3 #parse(Int64, ARGS[4])
const TASK_LABEL = ARGS[4]

using Bloqade
using Yao
using Printf
using Optim
using Plots
using Random
using DifferentialEquations
using Statistics: mean
using DataFrames
using CSV
using JLD2: jldsave

# using MultivariateStats: PCA
using Distributions: Normal, MvNormal
using Base.Iterators: partition
using Flux.Losses: binarycrossentropy
using Flux: gradient

using MLJ
using StatsBase: UnitRangeTransform, fit, transform

using MLDatasets: Iris, MNIST, FashionMNIST
# using OpenML

include("votingSmallRangeutils.jl")

println("Dataset type: $datasettype")
println("Lattice type: $LATTICE_TYPE")
println("Lattice Constant: $LATTICE_CONSTANT")


bce_grads(x, l) = gradient((a)->binarycrossentropy(a, l), x)[1]

if datasettype == "iris"
    df = Iris(as_df=true, dir="data/iris").dataframe
    # subset!(df, :class => x-> (x .== "Iris-setosa" .|| x .== "Iris-virginica") ) # train test 100/100
    # subset!(df, :class => x-> (x .== "Iris-versicolor" .|| x .== "Iris-virginica") ) # train test 100/100
    subset!(df, :class => x-> (x .== "Iris-setosa" .|| x .== "Iris-versicolor") )
    # transform!(df, ["sepallength"] .=> x-> x .* 2*pi ./ maximum(x))

    # DataFrames.select!(df, :sepallength => x-> x .* 2*pi ./ maximum(x), :sepalwidth => x-> x .* 2*pi ./ maximum(x), :petallength, :petalwidth , :class)
    # DataFrames.select!(df, :sepalwidth => x-> x .* 2*pi ./ maximum(x), :sepallength => x-> x .* 2*pi ./ maximum(x), :petallength, :petalwidth , :class)
    DataFrames.select!(df,:petallength => x-> x ./ maximum(x), :sepallength => x-> x ./ maximum(x), :sepalwidth=> x-> x .* 2*pi ./ maximum(x) , :petalwidth => x-> x .* 2*pi ./ maximum(x), :class)

    numTrainSamples = 70
    numTestSamples = 30

    batchSize = 70
    shuffle!(df)

    train_data = Matrix(df[1:numTrainSamples, Not(:class)])
    test_data = Matrix(df[numTrainSamples+1:end, Not(:class)])

    train_labels = df[1:numTrainSamples, :class] .== "Iris-setosa"
    test_labels = df[numTrainSamples+1:end, :class] .== "Iris-setosa"


elseif datasettype == "diabetes"
    
    const class0label = 0
    const class1label = 1
    
    # df = DataFrame(OpenML.load(37)) # something broke this entry point, so manually load the data instead
    df = DataFrame(CSV.File("data/diabetes/diabetes.csv"))

    DataFrames.select!(df, Not(:Outcome), :Outcome => x-> Float64.(x .== 1))
    
    filter!(row -> row.Glucose != 0 && row.BloodPressure != 0 && row.SkinThickness != 0 && row.BMI != 0 && row.Insulin != 0, df)

    labels, features = unpack(df, ==(:Outcome_function), shuffle=true, rng=MersenneTwister(0))

    numTotalPoints = length(labels)
    # numSubsample = min(TRAIN_DATA_PERCENTAGE/100 * numTotalPoints, 100) # subsample up to 100 points
    # numSubsample = TRAIN_DATA_PERCENTAGE/100 * numTotalPoints

    trainingRatio = TRAIN_DATA_PERCENTAGE / 100

    (full_train_data, full_test_data), (train_labels, test_labels) = MLJ.partition((features, labels), trainingRatio, shuffle=false, multi=true)

    PCA_model =  MLJ.@load PCA pkg=MultivariateStats
    pca_transform = machine(PCA_model(maxoutdim=5, variance_ratio=1.0), full_train_data)
    MLJ.fit!(pca_transform)

    scaled_train_data = MLJ.transform(pca_transform, full_train_data)
    scaled_test_data = MLJ.transform(pca_transform, full_test_data)

    # Standardizer = @load Standardizer pkg=MLJModels
    
    # standard_transform = machine(Standardizer(), scaled_train_data)
    # fit!(standard_transform)

    # unscale_train_data = MLJ.transform(standard_transform, scaled_train_data)    
    # unscale_test_data = MLJ.transform(standard_transform, scaled_test_data)

    # DataFrames.select!(unscale_train_data, All() .=> x -> x .* pi)
    # DataFrames.select!(unscale_test_data, All() .=> x -> x .* pi)

    # train_data = Matrix(unscale_train_data)
    # test_data = Matrix(unscale_test_data)
    
    maxVals = maximum.(eachcol(scaled_train_data))

    eachcol(scaled_train_data)

    # DataFrames.select!(scaled_train_data, :x1 -> x ./ maxVals[1], :x2 -> x / maxVals[2], :x3 -> x / maxVals[3], :x4 -> x / maxVals[4] )
    # DataFrames.select!(scaled_test_data, All() .=> x -> x .* pi)

    minmaxScaler = fit(UnitRangeTransform, Matrix(scaled_train_data), dims=1)

    # train_data = Matrix(scaled_train_data)
    # test_data = Matrix(scaled_test_data)

    train_data = transform(minmaxScaler, Matrix(scaled_train_data))
    test_data = transform(minmaxScaler, Matrix(scaled_test_data))

    train_data[:,1] = train_data[:,1] * (2*pi - pi/2) .+ pi/2
    train_data[:,2] = train_data[:,2] * (2*pi - pi/2) .+ pi/2
    train_data[:,5] = train_data[:,5] * -1 * (2*pi - pi/2) .- pi/2

    test_data[:,1] = test_data[:,1] * (2*pi - pi/2) .+ pi/2
    test_data[:,2] = test_data[:,2] * (2*pi - pi/2) .+ pi/2
    test_data[:,5] = test_data[:,5] * -1 * (2*pi - pi/2) .- pi/2

    numTrainSamples = size(train_data)[1]
    numTestSamples = size(test_data)[1]
    # batchSize = 64
    batchSize = numTrainSamples

elseif datasettype == "MNIST"

    const TRAIN_DATA_PERCENTAGE = 8

    if TASK_LABEL == "01"
        const class0label = 0
        const class1label = 1
    elseif TASK_LABEL == "34"
        const class0label = 3
        const class1label = 4
    elseif TASK_LABEL == "49"
        const class0label = 4
        const class1label = 9
    else
        exit("Incorrect task specification")
    end

    mnist_set = DataFrame(MNIST(; Tx=Float32, dir="data/MNIST"))
    mnist_test_set = DataFrame(MNIST(; Tx=Float32, split=:test, dir="data/MNIST"))
    
    subset!(mnist_set, :targets => x -> (x .== class0label .|| x .== class1label) )
    subset!(mnist_test_set, :targets => x -> (x .== class0label .|| x .== class1label) )

    DataFrames.select!(mnist_set, :features => x -> vec.(x), :targets)
    DataFrames.select!(mnist_test_set, :features => x -> vec.(x), :targets)

    DataFrames.select!(mnist_set, :features_function => identity => ["p$(i)" for i in 1:28*28], :targets)
    DataFrames.select!(mnist_test_set,  :features_function => identity => ["p$(i)" for i in 1:28*28], :targets)

    train_labels, train_features = unpack(mnist_set, ==(:targets), shuffle=true, rng=MersenneTwister(0))
    test_labels, test_features = unpack(mnist_test_set, ==(:targets), shuffle=true, rng=MersenneTwister(0))

    train_labels = Float64.(train_labels .== class1label)
    test_labels = Float64.(test_labels .== class1label)

    trainingRatio = TRAIN_DATA_PERCENTAGE / 100 * ( length(train_labels) + length(test_labels) ) / length(train_labels)

    (full_train_data, _), (train_labels, _) = MLJ.partition((train_features, train_labels), trainingRatio, shuffle=false, multi=true)

    full_test_data = test_features

    PCA_model =  MLJ.@load PCA pkg=MultivariateStats
    pca_transform = machine(PCA_model(maxoutdim=5, variance_ratio=1.0), full_train_data)
    MLJ.fit!(pca_transform)


    scaled_train_data = MLJ.transform(pca_transform, full_train_data)
    scaled_test_data = MLJ.transform(pca_transform, full_test_data)



    minmaxScaler = fit(UnitRangeTransform, Matrix(scaled_train_data), dims=1)

    train_data = transform(minmaxScaler, Matrix(scaled_train_data))
    test_data = transform(minmaxScaler, Matrix(scaled_test_data))

    train_data[:,1] = train_data[:,1] * (2*pi - pi/2) .+ pi/2
    train_data[:,2] = train_data[:,2] * (2*pi - pi/2) .+ pi/2
    # train_data[:,3] = train_data[:,3] * 2*pi
    # train_data[:,4] = train_data[:,4] * 2*pi
    train_data[:,5] = train_data[:,5] * -1 * (2*pi - pi/2) .- pi/2

    test_data[:,1] = test_data[:,1] * (2*pi - pi/2) .+ pi/2
    test_data[:,2] = test_data[:,2] * (2*pi - pi/2) .+ pi/2
    test_data[:,5] = test_data[:,5] * -1 * (2*pi - pi/2) .- pi/2

    numTrainSamples = size(train_data)[1]
    numTestSamples = size(test_data)[1]
    # batchSize = 64
    batchSize = numTrainSamples

elseif datasettype == "Fashion"

    const TRAIN_DATA_PERCENTAGE = 8

    if TASK_LABEL == "trouserboot"
        const class0label = 1
        const class1label = 9
    elseif TASK_LABEL == "bagsandal"
        const class0label = 8
        const class1label = 5
    elseif TASK_LABEL == "pullovercoat"
        const class0label = 2
        const class1label = 4
    else
        exit("Incorrect task specification")
    end

    mnist_set = DataFrame(FashionMNIST(; Tx=Float32, dir="data/FashionMNIST"))
    mnist_test_set = DataFrame(FashionMNIST(; Tx=Float32, split=:test, dir="data/FashionMNIST"))
    
    subset!(mnist_set, :targets => x -> (x .== class0label .|| x .== class1label) )
    subset!(mnist_test_set, :targets => x -> (x .== class0label .|| x .== class1label) )

    DataFrames.select!(mnist_set, :features => x -> vec.(x), :targets)
    DataFrames.select!(mnist_test_set, :features => x -> vec.(x), :targets)

    DataFrames.select!(mnist_set, :features_function => identity => ["p$(i)" for i in 1:28*28], :targets)
    DataFrames.select!(mnist_test_set,  :features_function => identity => ["p$(i)" for i in 1:28*28], :targets)

    train_labels, train_features = unpack(mnist_set, ==(:targets), shuffle=true, rng=MersenneTwister(0))
    test_labels, test_features = unpack(mnist_test_set, ==(:targets), shuffle=true, rng=MersenneTwister(0))

    train_labels = Float64.(train_labels .== class1label)
    test_labels = Float64.(test_labels .== class1label)

    trainingRatio = TRAIN_DATA_PERCENTAGE / 100 * ( length(train_labels) + length(test_labels) ) / length(train_labels)

    (full_train_data, _), (train_labels, _) = MLJ.partition((train_features, train_labels), trainingRatio, shuffle=false, multi=true)

    full_test_data = test_features

    PCA_model =  MLJ.@load PCA pkg=MultivariateStats
    pca_transform = machine(PCA_model(maxoutdim=5, variance_ratio=1.0), full_train_data)
    MLJ.fit!(pca_transform)


    scaled_train_data = MLJ.transform(pca_transform, full_train_data)
    scaled_test_data = MLJ.transform(pca_transform, full_test_data)



    minmaxScaler = fit(UnitRangeTransform, Matrix(scaled_train_data), dims=1)

    train_data = transform(minmaxScaler, Matrix(scaled_train_data))
    test_data = transform(minmaxScaler, Matrix(scaled_test_data))

    train_data[:,1] = train_data[:,1] * (2*pi - pi/2) .+ pi/2
    train_data[:,2] = train_data[:,2] * (2*pi - pi/2) .+ pi/2
    train_data[:,5] = train_data[:,5] * -1 * (2*pi - pi/2) .- pi/2

    test_data[:,1] = test_data[:,1] * (2*pi - pi/2) .+ pi/2
    test_data[:,2] = test_data[:,2] * (2*pi - pi/2) .+ pi/2
    test_data[:,5] = test_data[:,5] * -1 * (2*pi - pi/2) .- pi/2

    numTrainSamples = size(train_data)[1]
    numTestSamples = size(test_data)[1]
    # batchSize = 64
    batchSize = numTrainSamples

end

println("Num training points: $numTrainSamples")

function lossFuncWithGrad!(F, G, params, batch, labels)
    numAtoms = 4
    batchSize = length(labels)

    pΩ = params[1:NUM_TIERS*2]
    pΔg = params[NUM_TIERS*2+1:4*NUM_TIERS]
    pϕ = 0
    pΔl = params[4*NUM_TIERS+1:6*NUM_TIERS]
    hparam = params[6*NUM_TIERS+1:6*NUM_TIERS+div(numAtoms, 2)]

    bint = 20
    # latticeType = LATTICE_TYPE
    
    latticeConstant = 10
    numTiers = NUM_TIERS
    # observable = numq -> put(numq, 1=>projector(product_state(bit"1")))
    if measurement_scheme == "all"
        observable = numq -> 1/4 * (put(numq, 1=>projector(product_state(bit"1")) ) + put(numq, 2=>projector(product_state(bit"1"))) + put(numq, 3=>projector(product_state(bit"1")) ) + put(numq, 4=>projector(product_state(bit"1")) )  )
    elseif measurement_scheme == "half"
        observable = numq -> 1/2 * (put(numq, 1=>projector(product_state(bit"1")) ) + put(numq, 3=>projector(product_state(bit"1")) ) )
    else
        error("incorrect observable selection. Do \"all\" or \"half\"")
    end
    
    batch_logits = ones(batchSize)
    # println("length params ", length(params))
    batched_grad = zeros(length(params))

    for i in 1:batchSize

        hinput = batch[i, 3:4]
        Ω0 = batch[i, 2]
        Δg0 = batch[i, 1]
        ϕ0 = 0
        # Δl0 = -batch[i, 4]
        Δl0 = batch[i, 5]

        q_result, q_grad = residual_computation(hinput, Ω0, Δg0, ϕ0, Δl0, pΩ, pΔg, pϕ, pΔl, hparam, bint, LATTICE_TYPE, numAtoms, LATTICE_CONSTANT, numTiers, observable; computeGrad=(G !== nothing))

        # println("length qgrad ", length(q_grad))

        batch_logits[i] = q_result
        batched_grad += bce_grads(q_result, labels[i]) .* q_grad
    end

    println(batched_grad / batchSize)

    if G !== nothing
        G .= batched_grad / batchSize
    end

    if F !== nothing   
        return binarycrossentropy(batch_logits, labels)
    end

end

function runTraining(datasettype, LATTICE_TYPE, LATTICE_CONSTANT, train_data, test_data, train_labels, test_labels)
    NUM_ATOMS = 4
    batched_data = [train_data[r, :] for r in partition(1:numTrainSamples, batchSize)]
    batched_labels = [train_labels[r] for r in partition(1:numTrainSamples, batchSize)]

    numParams = NUM_TIERS * 2 * 3 + div(NUM_ATOMS,2)
    ## Run training
    init_guess = [1.0 for _ in 1:numParams]
    # solver = AdaMax(alpha=0.005)
    solver = AdaMax(alpha=0.005, beta_mean=0.8, beta_var=0.99)
    numIterations = 25

    trained_guess, training_loss = trainQNet(lossFuncWithGrad!, init_guess, batched_data, batched_labels, 1, 25, solver)

    trained_labels, trained_logits = inferQNet(train_data, trained_guess)
    pred_labels, pred_logits = inferQNet(test_data, trained_guess)

    # open("simplerunsVersicolorVirginica/chain_spacing_$(scaleFactor).txt","a") do io
    #     println(io, "Trained params: ", trained_guess)
    #     println(io, "Training losses: ", training_loss)
    #     println(io, "Training acc: ", sum(trained_labels .== train_labels) / numTrainSamples)
    #     println(io, "Testing acc: ", sum(pred_labels .== test_labels) / numTestSamples)
    # end
    fscore = FScore(levels=[0,1])

    training_accuracy = sum(trained_labels .== train_labels) / numTrainSamples
    testing_accuracy =  sum(pred_labels .== test_labels) / numTestSamples
    f1_score = fscore(pred_labels, test_labels)

    println("Results after $numIterations iterations")
    println("Trained params: ", trained_guess)
    println("Training losses: ", training_loss)
    println("Training acc: ", training_accuracy)
    println("Testing acc: ", testing_accuracy)
    println("f1score: ", f1_score)


    for idx in [1, 2]
        if true
            trained_guess, training_loss = trainQNet(lossFuncWithGrad!, trained_guess, batched_data, batched_labels, 1, 25, solver)

            trained_labels, trained_logits = inferQNet(train_data, trained_guess)
            pred_labels, pred_logits = inferQNet(test_data, trained_guess)

            training_accuracy = sum(trained_labels .== train_labels) / numTrainSamples
            testing_accuracy = sum(pred_labels .== test_labels) / numTestSamples
            f1_score = fscore(pred_labels, test_labels)

            println("Results after " * string(idx * 25 + 25) * " iterations")
            println("Trained params: ", trained_guess)
            println("Training losses: ", training_loss)
            println("Training acc: ", training_accuracy)
            println("Testing acc: ", testing_accuracy)
            println("f1score: ", f1_score)
        end
    end

    jldsave("votingsmallrangeresults/" * datasettype * measurement_scheme * "$(class0label)$(class1label)" * "_" * LATTICE_TYPE * "_" * string(Int(LATTICE_CONSTANT)) *"dataPercent"* string(Int(TRAIN_DATA_PERCENTAGE)) * "numTiers$(NUM_TIERS)" * ".jld2", trained_params=trained_guess, training_losses=training_loss, train_acc=training_accuracy, test_acc=testing_accuracy, f1=f1_score)
end

runTraining(datasettype, LATTICE_TYPE, LATTICE_CONSTANT, train_data, test_data, train_labels, test_labels)
