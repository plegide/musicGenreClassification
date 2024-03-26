using Flux, Flux.Losses, DelimitedFiles, Plots, ProgressMeter, Random, Statistics

function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    numClasses = length(classes)
    @assert(numClasses > 1)
    if numClasses == 2
        # Si solo hay dos clases, se devuelve una matriz con una columna
        feature = reshape(feature .== classes[1], :, 1)
    else
        # Si hay mas clases devuelve true en la columna en la que se corresponde con su clase 
        oneHot = Array{Bool,2}(undef, length(feature), numClasses)
        for numClass = 1:numClasses
            oneHot[:, numClass] .= (feature .== classes[numClass])
        end
        feature = oneHot
    end
    return feature
end;

function crossvalidation(N::Int64, k::Int64)
    ordered_vector = collect(1:k)
    repeated_vector = repeat(ordered_vector, Int(ceil(N / k)))
    first_N_values = repeated_vector[1:N]
    shuffled_vector = shuffle!(first_N_values)
    return shuffled_vector
end

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    indices = collect(1:length(targets))
    indices[targets] = crossvalidation(sum(targets), k)
    indices[.!targets] = crossvalidation(sum(.!targets), k)
    return indices
end

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    indices = zeros(Int, size(targets, 1))
    for i in 1:size(targets, 2)
        indices[findall(targets[:, i])] = crossvalidation(sum(targets[:, i]), k)
    end
    return indices
end

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    indices = crossvalidation(oneHotEncoding(targets, unique(targets)), k)
    return indices
end

function ANNCrossValidation(topology::AbstractArray{<:Int,1}, 
inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1}, 
crossValidationIndices::Array{Int64,1}; 
numExecutions::Int=50, 
transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, 
validationRatio::Real=0, maxEpochsVal::Int=20)
    numClasses = size(targets, 2)
    accuracies = zeros(numExecutions)
    for execution in 1:numExecutions
        # Separar el conjunto de datos en entrenamiento y validación
        trainingIndices = findall(.~(crossValidationIndices .== 1))
        validationIndices = findall(crossValidationIndices .== 1)
        trainingInputs = inputs[trainingIndices, :]
        validationInputs = inputs[validationIndices, :]
        trainingTargets = targets[trainingIndices, :]
        validationTargets = targets[validationIndices, :]
        
        # Entrenar la RNA
        (ann, losses, valLoss, valLossEpoch) = trainClassANN(topology, (trainingInputs', trainingTargets'), validationDataset=(validationInputs', validationTargets'), maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal)
        
        # Evaluar la RNA
        testOutputs = ann(validationInputs')
        testOutputs = testOutputs .> 0.5
        accuracies[execution] = accuracy(testOutputs, validationTargets)
    end
    return accuracies
end

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

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    return outputs .≥ threshold
end

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
    # ann = buildClassANN(size(inputs, 2), topology, size(targets, 2), transferFunctions=transferFunctions)
    # Definicion de la topologia de la ann
    #ann = buildClassANN(size(inputs', 2), topology, size(targets', 2), transferFunctions=transferFunctions)
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
                # Este if se usa para devolver la mejor ann del entrenamiento
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
    # ann = buildClassANN(size(inputs, 2), topology, 1, transferFunctions=transferFunctions)
    # Definicion de la topologia de la ann
    #ann = buildClassANN(size(inputs', 2), topology, size(targets', 2), transferFunctions=transferFunctions)
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
                # Este if se usa para devolver la mejor ann del entrenamiento
                bestValLoss = valLoss
                bestValLossEpoch = epoch
                bestAnn = ann
            end
        end
    end
    return bestAnn, losses, bestValLoss, bestValLossEpoch
end




 #=
function trainClassANN(topology::AbstractArray{<:Int,1}, 
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; 
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, 
    maxEpochsVal::Int=20)
=#

#=
function trainClassANN(topology::AbstractArray{<:Int,1}, 
trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; 
validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), 
testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), 
transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, 
maxEpochsVal::Int=20)
=#




function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    VP = sum(outputs .& targets)
    VN = sum((.~outputs) .& (.~targets))
    FP = sum(outputs .& (.~targets))
    FN = sum((.~outputs) .& targets)
    return (VP=VP, VN=VN, FP=FP, FN=FN)
end

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    outputs_bool = classifyOutputs(outputs, threshold=threshold)
    return confusionMatrix(outputs_bool, targets)
end

function printConfusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    (VP, VN, FP, FN) = confusionMatrix(outputs, targets)
    println("Matriz de confusión:")
    println("VP: $VP")
    println("VN: $VN")
    println("FP: $FP")
    println("FN: $FN")

    println("Precisión: $(accuracy(outputs, targets))")
end

function printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    outputs_bool = classifyOutputs(outputs, threshold=threshold)
    printConfusionMatrix(outputs_bool, targets)
end


#=
function calculateMetrics()
    # Calculamos el loss en entrenamiento y test. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
    trainingLoss = loss(trainingInputs', trainingTargets');
    validationLoss = loss(validationInputs', validationTargets');
    testLoss = loss(testInputs', testTargets');
    # Calculamos la salida de la RNA en entrenamiento y test. Para ello hay que pasar la matriz de entradas traspuesta (cada patron en una columna). La matriz de salidas tiene un patron en cada columna
    trainingOutputs = ann(trainingInputs');
    validationOutputs = ann(validationInputs');
    testOutputs = ann(testInputs');
    # Para calcular la precision, ponemos 2 opciones aqui equivalentes:
    # Pasar las matrices con los datos en las columnas. La matriz de salidas ya tiene un patron en cada columna
    trainingAcc = accuracy(trainingOutputs, 
   Array{Bool,2}(trainingTargets'); dataInRows=false);
    validationAcc = accuracy(validationOutputs,
   Array{Bool,2}(validationTargets'); dataInRows=false);
    testAcc = accuracy(testOutputs, Array{Bool,2}(testTargets');
    dataInRows=false);
    # Pasar las matrices con los datos en las filas. Hay que trasponer la matriz de salidas de la RNA, puesto que cada dato esta en una fila
    trainingAcc = accuracy(Array{Float64,2}(trainingOutputs'), 
   trainingTargets; dataInRows=true);
    validationAcc = accuracy(Array{Float64,2}(validationOutputs'), 
   validationTargets; dataInRows=true);
    testAcc = accuracy(Array{Float64,2}(testOutputs'), 
   testTargets; dataInRows=true);
    # Mostramos por pantalla el resultado de este ciclo de entrenamiento si nos lo han indicado
    if showText
    println("Epoch ", numEpoch, ": Training loss: ", trainingLoss, ", 
   accuracy: ", 100*trainingAcc, " % - Validation loss: ", validationLoss, ", 
   accuracy: ", 100*validationAcc, " % - Test loss: ", testLoss, ", accuracy: ", 
   100*testAcc, " %");
    end;
    return (trainingLoss, trainingAcc, validationLoss, validationAcc, 
   testLoss, testAcc)
    end;
   =#


function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, 
    inputs::Array{Float32,2}, targets::Array{Any,1}, numFolds::Int64)
    
        # Comprobamos que el numero de patrones coincide
        @assert(size(inputs,1)==length(targets));
    
        # Que clases de salida tenemos
        # Es importante calcular esto primero porque se va a realizar codificacion one-hot-encoding varias veces, y el orden de las clases deberia ser el mismo siempre
        classes = unique(targets);
    
        # Primero codificamos las salidas deseadas en caso de entrenar RR.NN.AA.
        if modelType==:ANN
        targets = oneHotEncoding(targets, classes);
        end;
    
        # Creamos los indices de crossvalidation
        crossValidationIndices = crossvalidation(size(inputs,1), numFolds);
    
        # Creamos los vectores para las metricas que se vayan a usar
        # En este caso, solo voy a usar precision y F1, en otro problema podrían ser distintas
        testAccuracies = Array{Float64,1}(undef, numFolds);
        testF1 = Array{Float64,1}(undef, numFolds);
    
        # Para cada fold, entrenamos
        for numFold in 1:numFolds
    
            # Si vamos a usar unos de estos 3 modelos
            if (modelType==:SVM) || (modelType==:DecisionTree) || (modelType==:kNN)
    
                # Dividimos los datos en entrenamiento y test
                trainingInputs = inputs[crossValidationIndices.!=numFold,:];
                testInputs = inputs[crossValidationIndices.==numFold,:];
                trainingTargets = targets[crossValidationIndices.!=numFold];
                testTargets = targets[crossValidationIndices.==numFold];
    
                if modelType==:SVM
                    model = SVC(kernel=modelHyperparameters["kernel"], 
                    degree=modelHyperparameters["kernelDegree"], 
                    gamma=modelHyperparameters["kernelGamma"], C=modelHyperparameters["C"]);
    
                elseif modelType==:DecisionTree
                    model = DecisionTreeClassifier(max_depth=modelHyperparameters["maxDepth"], random_state=1);
                elseif modelType==:kNN
                    model = KNeighborsClassifier(modelHyperparameters["numNeighbors"]);
                end;
     
               # Entrenamos el modelo con el conjunto de entrenamiento
                model = fit!(model, trainingInputs, trainingTargets);
    
                # Pasamos el conjunto de test
                testOutputs = predict(model, testInputs);
    
                # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
                (acc, _, _, _, _, _, F1, _) = confusionMatrix(testOutputs, testTargets);
    
            else
    
                # Vamos a usar RR.NN.AA.
                @assert(modelType==:ANN);
    
                # Dividimos los datos en entrenamiento y test
                trainingInputs = inputs[crossValidationIndices.!=numFold,:];
                testInputs = inputs[crossValidationIndices.==numFold,:];
                trainingTargets = targets[crossValidationIndices.!=numFold,:];
                testTargets = targets[crossValidationIndices.==numFold,:];
    
                # Como el entrenamiento de RR.NN.AA. es no determinístico, hay que entrenar varias veces, y
                # se crean vectores adicionales para almacenar las metricas para cada entrenamiento
                testAccuraciesEachRepetition = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);
                testF1EachRepetition = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);
    
                # Se entrena las veces que se haya indicado
                for numTraining in 1:modelHyperparameters["numExecutions"]
    
                    if modelHyperparameters["validationRatio"]>0
    
                        # Para el caso de entrenar una RNA con conjunto de validacion, hacemos una división adicional:
                        # dividimos el conjunto de entrenamiento en entrenamiento+validacion
                        # Para ello, hacemos un hold out
                        (trainingIndices, validationIndices) = holdOut(size(trainingInputs,1), modelHyperparameters["validationRatio"]*size(trainingInputs,1)/size(inputs,1));
                        # Con estos indices, se pueden crear los vectores finales que vamos a usar para entrenar una RNA
                        # Entrenamos la RNA, teniendo cuidado de codificar las salidas deseadas correctamente
                        #=
                        ann, = trainClassANN(modelHyperparameters["topology"],
                                (trainingInputs', trainingTargets'),
                                trainingInputs[validationIndices,:], 
                                trainingTargets[validationIndices,:],
                                testInputs, testTargets;
                                maxEpochs=modelHyperparameters["maxEpochs"], 
                                learningRate=modelHyperparameters["learningRate"], 
                                maxEpochsVal=modelHyperparameters["maxEpochsVal"]);
                                =#
                    else
    
                        # Si no se desea usar conjunto de validacion, se entrena unicamente con conjuntos de entrenamiento y test,
                        # teniendo cuidado de codificar las salidas deseadas correctamente
                        
                        ann, = trainClassANN(modelHyperparameters["topology"],
                            (trainingInputs', trainingTargets'),
                            validationDataset=(testInputs', testTargets'),
                            maxEpochs=modelHyperparameters["maxEpochs"], 
                            learningRate=modelHyperparameters["learningRate"]);
                            
                    end;
    
                    # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
                    (testAccuraciesEachRepetition[numTraining], _, _, _, _, _, testF1EachRepetition[numTraining], _) = confusionMatrix(collect(ann(testInputs')'), testTargets);
                end;
    
                # Calculamos el valor promedio de todos los entrenamientos de este fold
    
                acc = mean(testAccuraciesEachRepetition);
                F1 = mean(testF1EachRepetition);
            end;
    
            # Almacenamos las 2 metricas que usamos en este problema
            testAccuracies[numFold] = acc;
            testF1[numFold] = F1;
    
            println("Results in test in fold ", numFold, "/", numFolds, ": accuracy:", 100*testAccuracies[numFold], " %, F1: ", 100*testF1[numFold], " %");
    
        end; # for numFold in 1:numFolds
    
        println(modelType, ": Average test accuracy on a ", numFolds, "-fold crossvalidation: ", 100*mean(testAccuracies), ", with a standard deviation of ",100*std(testAccuracies));
    
        println(modelType, ": Average test F1 on a ", numFolds, "-fold crossvalidation: ", 100*mean(testF1), ", with a standard deviation of ", 100*std(testF1));
    
        return (mean(testAccuracies), std(testAccuracies), mean(testF1), std(testF1));
end;