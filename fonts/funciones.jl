using Flux, DelimitedFiles, Plots, ProgressMeter, Random, Statistics, WAV, FFTW
using Flux
using Flux.Losses
using Flux: onehotbatch, onecold
using Flux: adjust!
function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    numClasses = length(classes)
    @assert(numClasses > 1)
    if numClasses == 2
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

function oneHotEncoding(feature::AbstractArray{<:Any,1})
    classes = unique(feature)
    return oneHotEncoding(feature, classes)
end

function oneHotEncoding(feature::AbstractArray{Bool,1})
    return reshape(feature, :, 1)
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

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    mean, std = normalizationParameters
    dataset .= (dataset .- mean) ./ std
end

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset)
    normalizeZeroMean(dataset, normalizationParameters)
end

function normalizeZeroMean( dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    mean, std = normalizationParameters
    return (dataset .- mean) ./ std
end

function normalizeZeroMean(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset)
    return normalizeZeroMean(dataset, normalizationParameters)
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
    outputs_bool = outputs .≥ threshold
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
    ann = Chain()
    numInputsLayer = numInputs

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

# Practica 3

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


function trainClassANN(topology::AbstractArray{<:Int,1}, trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
    inputs, targets = trainingDataset
    ann = buildClassANN(size(inputs', 2), topology, size(targets', 2), transferFunctions=transferFunctions)
    loss(model, x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y);
    opt_state = Flux.setup(Adam(learningRate), ann)
    train_losses = [] 
    validation_losses = []
    test_losses = []
    bestValLoss = Inf
    bestValLossEpoch = 0
    bestAnn = deepcopy(ann)
    counter = 0
    @showprogress for epoch in 1:maxEpochs
        Flux.train!(loss, ann, [(inputs, targets)], opt_state)
        train_loss = loss(ann, inputs, targets)
        test_loss = loss(ann, testDataset...)
        push!(train_losses, train_loss)
        push!(test_losses, test_loss)

        if validationDataset != (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0))
            validation_loss = loss(ann, validationDataset...)
            push!(validation_losses, validation_loss)

            if validation_loss ≤ minLoss
                println("Pérdida mínima alcanzada después de $epoch ciclos.")
                break
            end

            # Arriba se mira que no MEJORE en 20 veces, cuanto MAYOR sea el loss, peor es, abajo se updatea el loss
            if validation_loss >= bestValLoss
                counter += 1
                if counter % maxEpochsVal == 0
                    return bestAnn, train_losses, validation_losses, test_losses, bestValLoss, bestValLossEpoch
                end
            else
                counter = 0
                bestValLoss = validation_loss
                bestValLossEpoch = epoch
                bestAnn = deepcopy(ann)
            end
        end
    end
    return ann, train_losses, validation_losses, test_losses, bestValLoss, bestValLossEpoch
end

function trainClassANN(topology::AbstractArray{<:Int,1}, trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
    inputs, targets = trainingDataset
    ann = buildClassANN(size(inputs', 2), topology, size(targets', 2), transferFunctions=transferFunctions)
    loss(model, x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y);
    opt_state = Flux.setup(Adam(learningRate), ann)
    train_losses = []
    validation_losses = []
    test_losses = []
    bestValLoss = Inf
    bestValLossEpoch = 0
    bestAnn = ann
    @showprogress for epoch in 1:maxEpochs
        Flux.train!(loss, ann, [(inputs', targets')], opt_state)
        train_loss = loss(ann, inputs, targets)
        validation_loss = loss(ann, validationDataset...)
        test_loss = loss(ann, inputs, testDataset...)
        push!(train_losses, train_loss)
        push!(validation_losses, validation_loss)
        push!(test_losses, test_loss)
        if loss_value ≤ minLoss
            println("Pérdida mínima alcanzada después de $epoch ciclos.")
            break
        end
        if epoch % maxEpochsVal == 0
            if validation_loss < bestValLoss
                bestValLoss = validation_loss
                bestValLossEpoch = epoch
                bestAnn = deepcopy(ann)
            end
        end
    end
    return ann, train_losses, validation_losses, test_losses, bestValLoss, bestValLossEpoch
end


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
        (ann, train_losses, validation_losses, test_losses, valLoss, valLossEpoch) = trainClassANN(topology, (trainingInputs', trainingTargets'), validationDataset=(validationInputs', validationTargets'), maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal)

        # Evaluar la RNA
        testOutputs = ann(validationInputs')
        testOutputs = testOutputs .> 0.5
        accuracies[execution] = accuracy(testOutputs, validationTargets)
    end
    return accuracies
end



function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    @assert(length(outputs)==length(targets));
    acc = accuracy(outputs, targets);
    errorRate = 1. - acc;
    recall = mean( outputs[ targets]); # Sensibilidad
    specificity = mean(.!outputs[.!targets]); # Especificidad
    precision = mean( targets[ outputs]); # Valor predictivo positivo
    NPV = mean(.!targets[.!outputs]); # Valor predictivo negativo
    if isnan(recall) && isnan(precision) # Los VN son el 100% de los patrones
        recall = 1.;
        precision = 1.;
    elseif isnan(specificity) && isnan(NPV) # Los VP son el 100% de los patrones
        specificity = 1.;
        NPV = 1.;
    end;
    
    recall = isnan(recall) ? 0. : recall;
    specificity = isnan(specificity) ? 0. : specificity;
    precision = isnan(precision) ? 0. : precision;
    NPV = isnan(NPV) ? 0. : NPV;
    F1 = (recall==precision==0.) ? 0. : 2*(recall*precision)/(recall+precision);
    confMatrix = Array{Int64,2}(undef, 2, 2);
    confMatrix[1,1] = sum(.!targets .& .!outputs); # VN
    confMatrix[1,2] = sum(.!targets .& outputs); # FP
    confMatrix[2,1] = sum( targets .& .!outputs); # FN
    confMatrix[2,2] = sum( targets .& outputs); # VP
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix)
   end;


confusionMatrix(outputs::AbstractArray{Float64,1}, targets::AbstractArray{Bool,1}; 
   threshold::Float64=0.5) = confusionMatrix(Array{Bool,1}(outputs.>=threshold), 
   targets);
   
function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    @assert(size(outputs)==size(targets));
    numClasses = size(targets,2);
    @assert(numClasses!=2);
    if (numClasses==1)
        return confusionMatrix(outputs[:,1], targets[:,1]);
    else
        @assert(all(sum(outputs, dims=2).==1));
        recall = zeros(numClasses);
        specificity = zeros(numClasses);
        precision = zeros(numClasses);
        NPV = zeros(numClasses);
        F1 = zeros(numClasses);
        confMatrix = Array{Int64,2}(undef, numClasses, numClasses);
        numInstancesFromEachClass = vec(sum(targets, dims=1));
        for numClass in findall(numInstancesFromEachClass.>0)
            (_, _, recall[numClass], specificity[numClass], precision[numClass],
            NPV[numClass], F1[numClass], _) = confusionMatrix(outputs[:,numClass], 
            targets[:,numClass]);
        end;
        confMatrix = Array{Int64,2}(undef, numClasses, numClasses);
        for numClassTarget in 1:numClasses, numClassOutput in 1:numClasses
            confMatrix[numClassTarget, numClassOutput] = 
            sum(targets[:,numClassTarget] .& outputs[:,numClassOutput]);
        end;
        if weighted
            weights = numInstancesFromEachClass./sum(numInstancesFromEachClass);
            recall = sum(weights.*recall);
            specificity = sum(weights.*specificity);
            precision = sum(weights.*precision);
            NPV = sum(weights.*NPV);
            F1 = sum(weights.*F1);
        else
            numClassesWithInstances = sum(numInstancesFromEachClass.>0);
            recall = sum(recall)/numClassesWithInstances;
            specificity = sum(specificity)/numClassesWithInstances;
            precision = sum(precision)/numClassesWithInstances;
            NPV = sum(NPV)/numClassesWithInstances;
            F1 = sum(F1)/numClassesWithInstances;
        end;
        acc = accuracy(outputs, targets);
        errorRate = 1 - acc;
        return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix);
    end;
end;
    

function confusionMatrix(outputs::AbstractArray{Any,1}, targets::AbstractArray{Any,1}; 
    weighted::Bool=true)
    @assert(all([in(output, unique(targets)) for output in outputs]));
    classes = unique(targets);
    return confusionMatrix(oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes); weighted=weighted);
    end;

    confusionMatrix(outputs::Array{Float64,2}, targets::Array{Bool,2}; 
    weighted::Bool=true) = confusionMatrix(classifyOutputs(outputs), targets; weighted=weighted);
    confusionMatrix(outputs::Array{Float32,2}, targets::Array{Bool,2}; weighted::Bool=true) = confusionMatrix(convert(Array{Float64,2}, outputs), targets; weighted=weighted);
    printConfusionMatrix(outputs::Array{Float32,2}, targets::Array{Bool,2}; weighted::Bool=true) = printConfusionMatrix(convert(Array{Float64,2}, outputs), targets; weighted=weighted);

    function printConfusionMatrix(outputs::Array{Bool,2}, targets::Array{Bool,2}; weighted::Bool=true)
        (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets; weighted=weighted);
        numClasses = size(confMatrix,1);
        writeHorizontalLine() = (for i in 1:numClasses+1 print("--------") end; println(""); );
        writeHorizontalLine();
        print("\t| ");

        if (numClasses==2)
            println(" - \t + \t|");
        else
            print.("Cl. ", 1:numClasses, "\t| ");
        end;
        println("");
        writeHorizontalLine();
        
        for numClassTarget in 1:numClasses
            if (numClasses==2)
                print(numClassTarget == 1 ? " - \t| " : " + \t| ");
            else
                print("Cl. ", numClassTarget, "\t| ");
            end;
        print.(confMatrix[numClassTarget,:], "\t| ");
        println("");
        writeHorizontalLine();
        end;

        println("Accuracy: ", acc);
        println("Error rate: ", errorRate);
        println("Recall: ", recall);
        println("Specificity: ", specificity);
        println("Precision: ", precision);
        println("Negative predictive value: ", NPV);
        println("F1-score: ", F1);
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix);
    end;
    printConfusionMatrix(outputs::Array{Float64,2}, targets::Array{Bool,2}; weighted::Bool=true) = printConfusionMatrix(classifyOutputs(outputs), targets; weighted=weighted)
        

function printConfusionMatrixTable(testOutputs, testTargets, classes)
    num_classes = length(classes)
    confusion_matrix = zeros(Int, num_classes, num_classes)

    for i in 1:length(testOutputs)
        predicted_class = findfirst(classes .== testOutputs[i])
        true_class = findfirst(classes .== testTargets[i])
        confusion_matrix[true_class, predicted_class] += 1
    end
    
    println("Matriz de Confusión:")
    println("       | ", join(classes, " | "))
    println("-------|", repeat("-----|", num_classes))
    for i in 1:num_classes
        println(classes[i], "| ", join(confusion_matrix[i, :], " | "))
    end
end

function printANNConfusionMatrix(testOutputs, testTargets, classes)
    num_classes = length(classes)
    confusion_matrix = zeros(Int, num_classes, num_classes)
    for i in 1:num_classes:length(testOutputs)
        actual_class_index = findnext(testOutputs, i)
        predicted_class_index = findnext(testTargets, i)
        if actual_class_index !== nothing && predicted_class_index !== nothing
            confusion_matrix[actual_class_index % num_classes + 1, predicted_class_index % num_classes + 1] += 1
        end
    end
    println("Matriz de Confusión:")
    println("       | ", join(classes, " | "))
    println("-------|", repeat("-----|", num_classes))
    for i in 1:num_classes
        println(classes[i], "| ", join(confusion_matrix[i, :], " | "))
    end
end


function convert_to_binary_vector(outputs::Matrix{Float32})
    num_samples = size(outputs, 2)
    binary_vector = Bool[]
    for i in 1:num_samples
        column = outputs[:, i]
        max_index = argmax(column)
        binary_column = [j == max_index ? true : false for j in 1:length(column)]
        append!(binary_vector, binary_column)
    end
    return binary_vector
end



function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, 
    inputs::Array{Float32,2}, targets::Array{Any,1}, numFolds::Int64)
    @assert(size(inputs,1)==length(targets));
    classes = unique(targets);
    
    if modelType==:ANN
        targets = oneHotEncoding(targets, classes);
    end;
    
    crossValidationIndices = crossvalidation(size(inputs,1), numFolds);
    testAccuracies = Array{Float64,1}(undef, numFolds);
    testF1 = Array{Float64,1}(undef, numFolds);
    
    for numFold in 1:numFolds
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
     
            model = fit!(model, trainingInputs, trainingTargets);
            testOutputs = predict(model, testInputs);
            (acc, _, _, _, _, _, F1, _) = confusionMatrix(testOutputs, testTargets);   
            printConfusionMatrixTable(testOutputs, testTargets, classes)

        else
            @assert(modelType==:ANN);
            # Dividimos los datos en entrenamiento y test
            trainingInputs = inputs[crossValidationIndices.!=numFold,:];
            testInputs = inputs[crossValidationIndices.==numFold,:];
            trainingTargets = targets[crossValidationIndices.!=numFold,:];
            testTargets = targets[crossValidationIndices.==numFold,:];

            tT = nothing
            tO = nothing

            testAccuraciesEachRepetition = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);
            testF1EachRepetition = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);

            (trainingIndices, validationIndices) = holdOut(size(trainingInputs,1), modelHyperparameters["validationRatio"]*size(trainingInputs,1)/size(inputs,1));
            validationInputs = trainingInputs[validationIndices,:];
            validationTargets = trainingTargets[validationIndices,:];
            trainingInputs = trainingInputs[trainingIndices,:];
            trainingTargets = trainingTargets[trainingIndices,:];
            testInputsMatrix = Matrix{Float32}(testInputs[1, :]')     
            testTargetsMatrix = Matrix{Bool}(reshape(testTargets[1, :], 1, :))
    
            for numTraining in 1:modelHyperparameters["numExecutions"]
                if modelHyperparameters["validationRatio"]>0
                        
                    ann, = trainClassANN(modelHyperparameters["topology"],
                            (trainingInputs', trainingTargets'),
                            validationDataset=(validationInputs', validationTargets'),
                            testDataset=(testInputsMatrix', testTargetsMatrix'),
                            maxEpochs=modelHyperparameters["maxEpochs"],
                            learningRate=modelHyperparameters["learningRate"],
                            maxEpochsVal=modelHyperparameters["maxEpochsVal"]);
                else
                    testInputsMatrix = Matrix{Float32}(testInputs[1, :]')  # Convertir las entradas a tipo Float32
                    testTargetsMatrix = Matrix{Bool}(reshape(testTargets[1, :], 1, :)) 
                    ann, = trainClassANN(modelHyperparameters["topology"],
                            (trainingInputs', trainingTargets'),
                            # validationDataset=(testInputs', testTargets'),
                            testDataset=(testInputsMatrix', testTargetsMatrix'),
                            maxEpochs=modelHyperparameters["maxEpochs"], 
                            learningRate=modelHyperparameters["learningRate"],
                            maxEpochsVal=modelHyperparameters["maxEpochsVal"]);
                end;
                  
                testOutPut = ann(testInputs')

                if length(classes) == 2
                    testOutPut = testOutPut .> 0.5
                    testTargets = testTargets .> 0.5
                    testOutPut = [testOutPut[i] for i in 1:length(testOutPut)];
                    testTargets2 = [testTargets[i] for i in 1:length(testTargets)];    
                    (testAccuraciesEachRepetition[numTraining], _, _, _, _, _, testF1EachRepetition[numTraining], _) = confusionMatrix(vec(testOutPut), vec(testTargets2));

                else
                    testBinaryOutput = convert_to_binary_vector(testOutPut)
                    (testAccuraciesEachRepetition[numTraining], _, _, _, _, _, testF1EachRepetition[numTraining], _) = confusionMatrix(testBinaryOutput, vec(testTargets'));
                    tO = testBinaryOutput
                    tT = vec(testTargets')
                end;
            end;
                # Calculamos el valor promedio de todos los entrenamientos de este fold
                if(length(classes) != 2)
                    printANNConfusionMatrix(tO, tT, classes)
                end;
                acc = mean(testAccuraciesEachRepetition);
                F1 = mean(testF1EachRepetition);
        end;

        testAccuracies[numFold] = acc;
        testF1[numFold] = F1;
        println("Results in test in fold ", numFold, "/", numFolds, ": accuracy:", 100*testAccuracies[numFold], " %, F1: ", 100*testF1[numFold], " %");
    
    end; 
    
    println(modelType, ": Average test accuracy on a ", numFolds, "-fold crossvalidation: ", 100*mean(testAccuracies), ", with a standard deviation of ",100*std(testAccuracies));
    println(modelType, ": Average test F1 on a ", numFolds, "-fold crossvalidation: ", 100*mean(testF1), ", with a standard deviation of ", 100*std(testF1));
    return (mean(testAccuracies), std(testAccuracies), mean(testF1), std(testF1));
end;



function cargar_datos(ruta_datos)
    genres = readdir(ruta_datos)
    datos = []
    etiquetas = []
    for genre in genres
        if(genre == ".gitignore")
            continue
        end
        archivos = readdir(joinpath(ruta_datos, genre))
        for archivo in archivos
            ruta = joinpath(ruta_datos, genre, archivo)
            if isfile(ruta) && endswith(archivo, ".wav")
                push!(datos, ruta)
                push!(etiquetas, genre)
            end
        end
    end
    return datos, etiquetas
end;

function audioFft(audio_file_path::AbstractString)
    wav_data, Fs = wavread(audio_file_path)

    n = length(wav_data)
    senalFrecuencia = abs.(fft(wav_data));

    if (iseven(n))
        @assert(mean(abs.(senalFrecuencia[2:Int(n/2)] .- senalFrecuencia[end:-1:(Int(n/2)+2)]))<1e-8);
        senalFrecuencia = senalFrecuencia[1:(Int(n/2)+1)];
    else
        @assert(mean(abs.(senalFrecuencia[2:Int((n+1)/2)] .- senalFrecuencia[end:-1:(Int((n-1)/2)+2)]))<1e-8);
        senalFrecuencia = senalFrecuencia[1:(Int((n+1)/2))];
    end;
    return senalFrecuencia
end;


function preprocesar_datos(datos)
    espectrogramas = [audioFft(audio) for audio in datos]
    return espectrogramas
end;

criterioFin = false;
mejorPrecision = -Inf;
numCiclo = 0;
numCicloUltimaMejora = 0;
mejorModelo = nothing;
eta = 0.01;


function deepLearning()
    N = 291
    datos, generos = cargar_datos("segments");

    datos_procesados = preprocesar_datos(datos);
    println(typeof(datos_procesados))

    datos_procesados_matrix = reduce(vcat, [x' for x in datos_procesados])
    normalizeMinMax!(datos_procesados_matrix);
    

    (trainingIndices, testIndices) = holdOut(size(datos_procesados_matrix,1), 0.2);
    trainingInputs = [datos_procesados_matrix[i, :] for i in trainingIndices]
    testInputs = [datos_procesados_matrix[i, :] for i in testIndices]
    trainingTargets2 = generos[trainingIndices, :]
    testTargets = generos[testIndices, :]
    
    trainingTargets = oneHotEncoding(vec(trainingTargets2))
    testTargets = oneHotEncoding(vec(testTargets))
    
    inputs = Array{Float32, 3}(undef, size(first(trainingInputs), 1), 1, length(trainingInputs))
    for i in eachindex(trainingInputs)
        inputs[:, :, i] = reshape(trainingInputs[i], :, 1)
    end
    train_set = (inputs, trainingTargets)
    
    testInputArr = Array{Float32, 3}(undef, size(first(testInputs), 1), 1, length(testInputs))
    for i in eachindex(testInputs)
        testInputArr[:, :, i] = reshape(testInputs[i], :, 1)
    end
    test_set = (testInputArr, testTargets)



    println("Tamaño de la matriz de entrenamiento: ", size(inputs))
    println("   Longitud de la señal: ", size(inputs,1))
    println("   Numero de canales: ", size(inputs,2))
    println("   Numero de instancias: ", size(inputs,3))

    GC.gc()
    funcionTransferenciaCapasConvolucionales = tanh;
    # Definimos la red con la funcion Chain, que concatena distintas capas
    ann = Chain(
        Conv((6,), 1=>16, pad=1, funcionTransferenciaCapasConvolucionales),
        MaxPool((2,)),
        Conv((6,), 16=>32, pad=1, funcionTransferenciaCapasConvolucionales),
        MaxPool((2,)),
        Conv((6,), 32=>32, pad=1, funcionTransferenciaCapasConvolucionales),
        MaxPool((2,)),
        x -> reshape(x, :, size(x, 3)),
        Dense(130976, 7),
        softmax
    );


    L1 = 0.01;
    L2 = 0;
    absnorm(x) = sum(abs , x)
    sqrnorm(x) = sum(abs2, x)
    loss(model,x,y) = ((size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y)) + L1*sum(absnorm, Flux.params(model)) + L2*sum(sqrnorm, Flux.params(model));

    function accuracy(batch)  
        return mean(onecold(ann(batch[1])) .== onecold(batch[2]')); 
    end;

    println("Ciclo 0: Precision en el conjunto de entrenamiento: ", 100*accuracy(train_set) , " %");

    opt_state = Flux.setup(Adam(eta), ann);

    println("Comenzando entrenamiento...")

    while !criterioFin
        
        global numCicloUltimaMejora, numCiclo, mejorPrecision, mejorModelo, criterioFin;
        Flux.train!(loss, ann, [(train_set[1], train_set[2]')], opt_state);
        numCiclo += 1;
        precisionEntrenamiento = accuracy(train_set);
        println("Ciclo ", numCiclo, ": Precision en el conjunto de entrenamiento: ", 100*precisionEntrenamiento, " %");

        if (precisionEntrenamiento > mejorPrecision)
            mejorPrecision = precisionEntrenamiento;
            precisionTest = accuracy(test_set);
            println("   Mejora en el conjunto de entrenamiento -> Precision en el conjunto de test: ", 100*precisionTest, " %");
            mejorModelo = deepcopy(ann);
            numCicloUltimaMejora = numCiclo;
        end

        if (numCiclo - numCicloUltimaMejora >= 5) && (eta > 1e-6)
            global eta
            eta /= 10.0
            println("   No se ha mejorado la precision en el conjunto de entrenamiento en 5 ciclos, se baja la tasa de aprendizaje a ", eta);
            adjust!(opt_state, eta)
            numCicloUltimaMejora = numCiclo;
        end

        if (precisionEntrenamiento >= 0.999)
            println("   Se para el entenamiento por haber llegado a una precision de 99.9%")
            criterioFin = true;
        end

        if (numCiclo - numCicloUltimaMejora >= 10)
            println("   Se para el entrenamiento por no haber mejorado la precision en el conjunto de entrenamiento durante 10 ciclos")
            criterioFin = true;
        end
    end


end