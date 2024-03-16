using Flux, Flux.Losses, DelimitedFiles, Plots, ProgressMeter, Random, Statistics


function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    numClasses = length(classes)
    @assert(numClasses > 1)
    if numClasses == 2
        # Si solo hay dos clases, se devuelve una matriz con una columna
        feature = reshape(feature .== classes[1], :, 1)
    else
        oneHot = Array{Bool,2}(undef, length(feature), numClasses)
        for numClass = 1:numClasses
            oneHot[:, numClass] .= (feature .== classes[numClass])
        end
        feature = oneHot
    end
    return feature
end;

function calculateMinMaxNormalizationParameters(data::AbstractArray{<:Real,2})
    mins = minimum(data, dims=1)
    maxs = maximum(data, dims=1)
    return (mins, maxs)
end

function calculateZeroMeanNormalizationParameters(data::AbstractArray{<:Real,2})
    means = mean(data, dims=1)
    stds = std(data, dims=1)
    return (means, stds)
end

function holdOut(N::Int, P::Float64)
    @assert ((P>=0.) & (P<=1.));
    indices = randperm(N);
    numTrainingInstances = Int(round(N*(1-P)));
    return (indices[1:numTrainingInstances], indices[numTrainingInstances+1:end]);
end

function holdOut(N::Int, Pval::Float64, Ptest::Float64)
    @assert ((Pval>=0.) & (Pval<=1.));
    @assert ((Ptest>=0.) & (Ptest<=1.));
    @assert ((Pval+Ptest)<=1.);
    # Primero separamos en entrenamiento+validation y test
    (trainingValidationIndices, testIndices) = holdOut(N, Ptest);
    # Después separamos el conjunto de entrenamiento+validation
    (trainingIndices, validationIndices) = holdOut(length(trainingValidationIndices), Pval*N/length(trainingValidationIndices))
    return (trainingValidationIndices[trainingIndices], trainingValidationIndices[validationIndices], testIndices);
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    mins, maxs = normalizationParameters
    dataset .= (dataset .- mins) ./ (maxs .- mins)
end

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset)
    normalizeMinMax!(dataset, normalizationParameters)
end

function normalizeMinMax( dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    mins, maxs = normalizationParameters
    return (dataset .- mins) ./ (maxs .- mins)
end

function normalizeMinMax(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset)
    return normalizeMinMax(dataset, normalizationParameters)
end

function classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5)
    return outputs .≥ threshold
end

function classifyOutputs(outputs::Array{Float64,2}; dataInRows::Bool=true, 
threshold::Float64=0.5)
 numOutputs = size(outputs, dataInRows ? 2 : 1);
 @assert(numOutputs!=2)
 if numOutputs==1
 return convert(Array{Bool,2}, outputs.>=threshold);
 else
 # Miramos donde esta el valor mayor de cada instancia con la funcion 
findmax
 (_,indicesMaxEachInstance) = findmax(outputs, dims= dataInRows ? 2 : 1);
 outputsBoolean = Array{Bool,2}(falses(size(outputs)));
 outputsBoolean[indicesMaxEachInstance] .= true;
 # Comprobamos que efectivamente cada patron solo este clasificado en una
clase
 @assert(all(sum(outputsBoolean, dims=dataInRows ? 2 : 1).==1));
 return outputsBoolean;
 end;
end;

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    return mean(outputs .== targets)
end

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    if size(outputs, 2) == 1 && size(targets, 2) == 1
        return accuracy(outputs[:, 1], targets[:, 1])
    else
        return mean(all(outputs .== targets, dims=2))
    end
end

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    outputs_bool = classifyOutputs(outputs)
    return accuracy(outputs_bool, targets)
end

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    if size(outputs, 2) == 1 && size(targets, 2) == 1
        return accuracy(outputs[:, 1], targets[:, 1])
    else
        outputs_bool = classifyOutputs(outputs, threshold=threshold)
        return accuracy(outputs_bool, targets)
    end
end

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    # Crear una RNA vacía
    ann = Chain()

    # Crear una variable para el número de entradas de cada capa
    numInputsLayer = numInputs

    # Añadir capas ocultas
    for numOutputsLayer in topology
        # Obtener la función de transferencia para esta capa
        transferFunction = transferFunctions[isempty(transferFunctions) ? 1 : length(ann.layers) + 1]

        # Añadir la capa oculta a la RNA
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunction))

        # Actualizar el número de entradas para la próxima capa
        numInputsLayer = numOutputsLayer
    end

    # Añadir la capa de salida con la función de transferencia adecuada
    if numOutputs == 1
        # Si solo hay una neurona de salida, usar función sigmoid
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, σ))
    else
        # Si hay más de una neurona de salida, usar función softmax
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs), softmax)
    end

    return ann
end

function trainClassANN(topology::AbstractArray{<:Int,1}, trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
    inputs, targets = trainingDataset
    print(size(inputs), size(targets))
    # ann = buildClassANN(size(inputs, 2), topology, size(targets, 2), transferFunctions=transferFunctions)
    ann = Chain( 
    Dense(2, 4, σ), 
    Dense(4, 1, σ) );
    loss(model, x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y);
    opt_state = Flux.setup(Adam(0.01), ann) 
    losses = []  # Lista para almacenar los valores de pérdida en cada ciclo
    bestValLoss = Inf
    bestValLossEpoch = 0
    bestAnn = ann
    @showprogress for epoch in 1:maxEpochs
        Flux.train!(loss, ann, [(inputs, targets)], opt_state)
        loss_value = loss(ann, inputs, targets)
        push!(losses, loss_value)
        if loss_value ≤ minLoss
            println("Pérdida mínima alcanzada después de $epoch ciclos.")
            break
        end
        if epoch % maxEpochsVal == 0
            valLoss = loss(ann, validationDataset...)
            if valLoss < bestValLoss
                bestValLoss = valLoss
                bestValLossEpoch = epoch
                bestAnn = ann
            end
        end
    end
    return bestAnn, losses, bestValLoss, bestValLossEpoch
end

function trainClassANN(topology::AbstractArray{<:Int,1}, trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
    inputs, targets = trainingDataset
    ann = buildClassANN(size(inputs, 2), topology, 1, transferFunctions=transferFunctions)
    losses = []  # Lista para almacenar los valores de pérdida en cada ciclo
    loss(model, x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y);
    opt_state = Flux.setup(Adam(learningRate), ann)
    bestValLoss = Inf
    bestValLossEpoch = 0
    bestAnn = ann
    @showprogress for epoch in 1:maxEpochs
        Flux.train!(loss, ann, [(inputs', targets')], opt_state)
        loss_value = loss(ann, inputs', targets')
        push!(losses, loss_value)
        if loss_value ≤ minLoss
            println("Pérdida mínima alcanzada después de $epoch ciclos.")
            break
        end
        if epoch % maxEpochsVal == 0
            valLoss = loss(ann, validationDataset...)
            if valLoss < bestValLoss
                bestValLoss = valLoss
                bestValLossEpoch = epoch
                bestAnn = ann
            end
        end
    end
    return bestAnn, losses, bestValLoss, bestValLossEpoch
end

# 1. Cargar el conjunto de datos
dataset = readdlm("classicOrMetal.data",',');
inputs = dataset[:,1:2];
targets = dataset[:,3];
classes = unique(targets);

inputs = convert(Array{Float32,2},inputs);
targets = oneHotEncoding(targets, classes);

# 2. Separar el conjunto de datos en entrenamiento, validacion y prueba
(trainIndexes, validationIndexes, testIndexes) = holdOut(size(inputs,1), 0.2, 0.2);

trainInputs = inputs[trainIndexes,:];
validationInputs = inputs[validationIndexes,:];
testInputs = inputs[testIndexes,:];
trainTargets = targets[trainIndexes,:];
validationTargets = targets[validationIndexes,:];
testTargets = targets[testIndexes,:];

# 3. Calcular parametros de normalizacion
normalizationParams = calculateMinMaxNormalizationParameters(trainInputs);

# 4. Normalizar el conjunto de datos
normalizeMinMax!(trainInputs, normalizationParams);
normalizeMinMax!(validationInputs, normalizationParams);
normalizeMinMax!(testInputs, normalizationParams);

# 5. Entrenar la RNA
topology = [2, 1];

(ann, losses, valLoss, valLossEpoch) = trainClassANN(topology, (trainInputs', trainTargets'), validationDataset=(validationInputs', validationTargets'));
# 6. Evaluar la RNA
testLoss = loss(ann, testInputs', testTargets');
testOutputs = ann(testInputs');
testAccuracy = accuracy(testOutputs, testTargets);

# 7. Mostrar resultados
println("Pérdida de prueba: $testLoss")
println("Precisión de prueba: $testAccuracy")

plot(losses, label="Pérdida de entrenamiento")
xlabel!("Ciclo")
ylabel!("Pérdida")
title!("Pérdida de entrenamiento durante el entrenamiento de la RNA")
plot!([valLossEpoch], [valLoss], seriestype=:scatter, label="Pérdida de validación mínima")
plot!([valLossEpoch], [valLoss], seriestype=:vline, label="Pérdida de validación mínima")
plot!([valLossEpoch], [valLoss], seriestype=:hline, label="Pérdida de validación mínima")

# 8. Guardar la RNA

