using Flux, Flux.Losses, DelimitedFiles, Plots, ProgressMeter


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

function trainClassANN(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)

    inputs, targets = dataset

    # Crear una RNA usando la función buildClassANN
    ann = buildClassANN(size(inputs, 2), topology, size(targets, 2), transferFunctions=transferFunctions)

    losses = []  # Lista para almacenar los valores de pérdida en cada ciclo

    # Configurar el optimizador y el estado inicial
    opt_state = Flux.setup(Adam(learningRate), ann)

    # Bucle de entrenamiento
    @showprogress for epoch in 1:maxEpochs
        # Realizar una iteración de entrenamiento
        Flux.train!(loss, ann, [(inputs', targets')], opt_state)

        # Calcular la pérdida en esta época y almacenarla en la lista
        loss_value = loss(ann, inputs', targets')  # se tranponen las matrices por convencion de la RR.NN.AA
        push!(losses, loss_value)

        # Verificar si se ha alcanzado la pérdida mínima
        if loss_value ≤ minLoss
            println("Pérdida mínima alcanzada después de $epoch ciclos.")
            break
        end

    end
    return ann, losses
end


function trainClassANN(topology::AbstractArray{<:Int,1},
    (inputs, targets)::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)

    dataset = (inputs, reshape(targets, :, 1))
    return trainClassANN(topology, dataset, transferFunctions=transferFunctions,
        maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate)
end

dataset = readdlm("classicOrMetal.data",',');
inputs = dataset[:,1:2];
targets = dataset[:,3];
classes = unique(targets);

inputs = convert(Array{Float32,2},inputs);
targets = oneHotEncoding(targets, classes);

ann, losses = trainClassANN([2, 3, 2], (inputs, targets), maxEpochs=1000, minLoss=0.01, learningRate=0.01)