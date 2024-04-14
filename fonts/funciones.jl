using Flux, Flux.Losses, DelimitedFiles, Plots, ProgressMeter, Random, Statistics

#Práctica 2
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
    # ann = buildClassANN(size(inputs, 2), topology, size(targets, 2), transferFunctions=transferFunctions)
    # Definicion de la topologia de la ann
    ann = buildClassANN(size(inputs', 2), topology, size(targets', 2), transferFunctions=transferFunctions)
    # ann = Chain( 
    # Dense(2, 4, σ), 
    # Dense(4, 1, σ) );
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
    print("ESTE")
    inputs, targets = trainingDataset
    # ann = buildClassANN(size(inputs, 2), topology, 1, transferFunctions=transferFunctions)
    # Definicion de la topologia de la ann
    ann = buildClassANN(size(inputs', 2), topology, size(targets', 2), transferFunctions=transferFunctions)
    # ann = Chain( 
    # Dense(2, 4, σ), 
    # Dense(4, 1, σ) );
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
                # Este if se usa para devolver la mejor ann del entrenamiento
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

############################################################


function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    @assert(length(outputs)==length(targets));
    # Para calcular la precision y la tasa de error, se puede llamar a las funciones definidas en la practica 2
    acc = accuracy(outputs, targets); # Precision, definida previamente en una practica anterior
    errorRate = 1. - acc;
    recall = mean( outputs[ targets]); # Sensibilidad
    specificity = mean(.!outputs[.!targets]); # Especificidad
    precision = mean( targets[ outputs]); # Valor predictivo positivo
    NPV = mean(.!targets[.!outputs]); # Valor predictivo negativo
    # Controlamos que algunos casos pueden ser NaN
    # Para el caso de sensibilidad y especificidad, en un conjunto de entrenamiento estos no pueden ser NaN, porque esto indicaria que se ha intentadoentrenar con una unica clase
    # Sin embargo, sí pueden ser NaN en el caso de aplicar un modelo en un conjunto de test, si este sólo tiene patrones de una clase
    # Para VPP y VPN, sí pueden ser NaN en caso de que el clasificador lo haya clasificado todo como negativo o positivo respectivamente
    # En estos casos, estas metricas habria que dejarlas a NaN para indicar que no se han podido evaluar
    # Sin embargo, como es posible que se quiera combinar estos valores al evaluar una clasificacion multiclase, es necesario asignarles un valor. El criterio que se usa aqui es que estos valores seran igual a 0
    # Ademas, hay un caso especial: cuando los VP son el 100% de los patrones, olos VN son el 100% de los patrones

    # En este caso, el sistema ha actuado correctamente, así que controlamos primero este caso
    if isnan(recall) && isnan(precision) # Los VN son el 100% de los patrones
    recall = 1.;
    precision = 1.;
    elseif isnan(specificity) && isnan(NPV) # Los VP son el 100% de los patrones
    specificity = 1.;
    NPV = 1.;
    end;
    # Ahora controlamos los casos en los que no se han podido evaluar las metricas excluyendo los casos anteriores
    recall = isnan(recall) ? 0. : recall;
    specificity = isnan(specificity) ? 0. : specificity;
    precision = isnan(precision) ? 0. : precision;
    NPV = isnan(NPV) ? 0. : NPV;
    # Calculamos F1, teniendo en cuenta que si sensibilidad o VPP es NaN (pero no ambos), el resultado tiene que ser 0 porque si sensibilidad=NaN entonces VPP=0 y viceversa
    F1 = (recall==precision==0.) ? 0. : 
   2*(recall*precision)/(recall+precision);
    # Reservamos memoria para la matriz de confusion
    confMatrix = Array{Int64,2}(undef, 2, 2);
    # Ponemos en las filas los que pertenecen a cada clase (targets) y en las columnas los clasificados (outputs)
    # Primera fila/columna: negativos
    # Segunda fila/columna: positivos
    # Primera fila: patrones de clase negativo, clasificados como negativos o positivos
    confMatrix[1,1] = sum(.!targets .& .!outputs); # VN
    confMatrix[1,2] = sum(.!targets .& outputs); # FP
    # Segunda fila: patrones de clase positiva, clasificados como negativos o positivos
    confMatrix[2,1] = sum( targets .& .!outputs); # FN
    confMatrix[2,2] = sum( targets .& outputs); # VP
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix)
   end;


confusionMatrix(outputs::AbstractArray{Float64,1}, targets::AbstractArray{Bool,1}; 
   threshold::Float64=0.5) = confusionMatrix(Array{Bool,1}(outputs.>=threshold), 
   targets);
   

   function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; 
    weighted::Bool=true)
     @assert(size(outputs)==size(targets));
     numClasses = size(targets,2);
     # Nos aseguramos de que no hay dos columnas
     @assert(numClasses!=2);
     if (numClasses==1)
     return confusionMatrix(outputs[:,1], targets[:,1]);
     else
     # Nos aseguramos de que en cada fila haya uno y sólo un valor a true
     @assert(all(sum(outputs, dims=2).==1));
     # Reservamos memoria para las metricas de cada clase, inicializandolas a 0 porque algunas posiblemente no se calculen
     recall = zeros(numClasses);
     specificity = zeros(numClasses);
     precision = zeros(numClasses);
     NPV = zeros(numClasses);
     F1 = zeros(numClasses);
     # Reservamos memoria para la matriz de confusion
     confMatrix = Array{Int64,2}(undef, numClasses, numClasses);
     # Calculamos el numero de patrones de cada clase
     numInstancesFromEachClass = vec(sum(targets, dims=1));
     # Calculamos las metricas para cada clase, esto se haria con un bucle similar a "for numClass in 1:numClasses" que itere por todas las clases
     # Sin embargo, solo hacemos este calculo para las clases que tengan algun patron
     # Puede ocurrir que alguna clase no tenga patrones como consecuencia dehaber dividido de forma aleatoria el conjunto de patrones entrenamiento/test
     # En aquellas clases en las que no haya patrones, los valores de las metricas seran 0 (los vectores ya estan asignados), y no se tendran en cuenta a la hora de unir estas metricas
     for numClass in findall(numInstancesFromEachClass.>0)
     # Calculamos las metricas de cada problema binario correspondiente acada clase y las almacenamos en los vectores correspondientes
     (_, _, recall[numClass], specificity[numClass], precision[numClass],
    NPV[numClass], F1[numClass], _) = confusionMatrix(outputs[:,numClass], 
    targets[:,numClass]);
     end;
     # Reservamos memoria para la matriz de confusion
     confMatrix = Array{Int64,2}(undef, numClasses, numClasses);
     # Calculamos la matriz de confusión haciendo un bucle doble que itere sobre las clases
     for numClassTarget in 1:numClasses, numClassOutput in 1:numClasses
     # Igual que antes, ponemos en las filas los que pertenecen a cada clase (targets) y en las columnas los clasificados (outputs)
     confMatrix[numClassTarget, numClassOutput] = 
    sum(targets[:,numClassTarget] .& outputs[:,numClassOutput]);
     end;
     # Aplicamos las forma de combinar las metricas macro o weighted
     if weighted
     # Calculamos los valores de ponderacion para hacer el promedio
     weights = numInstancesFromEachClass./sum(numInstancesFromEachClass);
     recall = sum(weights.*recall);
     specificity = sum(weights.*specificity);
     precision = sum(weights.*precision);
     NPV = sum(weights.*NPV);
     F1 = sum(weights.*F1);
     else
     # No realizo la media tal cual con la funcion mean, porque puede haber clases sin instancias
     # En su lugar, realizo la media solamente de las clases que tengan instancias
     numClassesWithInstances = sum(numInstancesFromEachClass.>0);
     recall = sum(recall)/numClassesWithInstances;
     specificity = sum(specificity)/numClassesWithInstances;
     precision = sum(precision)/numClassesWithInstances;
     NPV = sum(NPV)/numClassesWithInstances;
     F1 = sum(F1)/numClassesWithInstances;
     end;
     # Precision y tasa de error las calculamos con las funciones definidas previamente
     acc = accuracy(outputs, targets);
     errorRate = 1 - acc;
     return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix);
     end;
    end;
    

    function confusionMatrix(outputs::AbstractArray{Any,1}, targets::AbstractArray{Any,1}; 
        weighted::Bool=true)
         # Comprobamos que todas las clases de salida esten dentro de las clases de las salidas deseadas
         @assert(all([in(output, unique(targets)) for output in outputs]));
         classes = unique(targets);
         # Es importante calcular el vector de clases primero y pasarlo como argumento a las 2 llamadas a oneHotEncoding para que el orden de las clases sea el mismo en ambas matrices
         return confusionMatrix(oneHotEncoding(outputs, classes), 
        oneHotEncoding(targets, classes); weighted=weighted);
        end;
        confusionMatrix(outputs::Array{Float64,2}, targets::Array{Bool,2}; 
        weighted::Bool=true) = confusionMatrix(classifyOutputs(outputs), targets; 
        weighted=weighted);
        # De forma similar a la anterior, añado estas funcion porque las RR.NN.AA. dan la salida como matrices de valores Float32 en lugar de Float64
        # Con estas funcion se pueden usar indistintamente matrices de Float32 o Float64
        confusionMatrix(outputs::Array{Float32,2}, targets::Array{Bool,2}; 
        weighted::Bool=true) = confusionMatrix(convert(Array{Float64,2}, outputs), 
        targets; weighted=weighted);
        printConfusionMatrix(outputs::Array{Float32,2}, targets::Array{Bool,2}; 
        weighted::Bool=true) = printConfusionMatrix(convert(Array{Float64,2}, outputs), 
        targets; weighted=weighted);
        # Funciones auxiliares para visualizar por pantalla la matriz de confusion y lasmetricas que se derivan de ella
        function printConfusionMatrix(outputs::Array{Bool,2}, targets::Array{Bool,2}; 
        weighted::Bool=true)
         (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = 
        confusionMatrix(outputs, targets; weighted=weighted);
         numClasses = size(confMatrix,1);
         writeHorizontalLine() = (for i in 1:numClasses+1 print("--------") end; 
        println(""); );
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
         # print.(confMatrix[numClassTarget,:], "\t");
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
         return (acc, errorRate, recall, specificity, precision, NPV, F1, 
        confMatrix);
        end;
        printConfusionMatrix(outputs::Array{Float64,2}, targets::Array{Bool,2}; 
        weighted::Bool=true) = printConfusionMatrix(classifyOutputs(outputs), targets; 
        weighted=weighted)
        





###########################################################3
#=
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
                #testOutputs = oneHotEncoding(testOutputs, unique(testOutputs))
                #testTargets = oneHotEncoding(testTargets, unique(testTargets))
                # bitvector to array of boolean 1 dimension
                 #testOutputs = [testOutputs[i] for i in 1:length(testOutputs)];
                 #testOutputs = convert(AbstractArray{Bool,1},testOutputs)
                 #testTargets = [testTargets[i] for i in 1:length(testTargets)];
                 #testTargets = convert(AbstractArray{Bool,1},testTargets)
                 
                 
                # print(typeof(testOutputs))
                # print(typeof(testTargets))
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
                        validationInputs = trainingInputs[validationIndices,:];
                        validationTargets = trainingTargets[validationIndices,:];
                        trainingInputs = trainingInputs[trainingIndices,:];
                        trainingTargets = trainingTargets[trainingIndices,:];
                        # Con estos indices, se pueden crear los vectores finales que vamos a usar para entrenar una RNA
                        # Entrenamos la RNA, teniendo cuidado de codificar las salidas deseadas correctamente
                        # ann, = trainClassANN(modelHyperparameters["topology"], trainingDataset=(trainingInputs[trainingIndices,:]', trainingTargets[trainingIndices,:]'), validationDataset=(trainingInputs[validationIndices,:]', trainingTargets[validationIndices,:]'), testDataset=(testInputs[1,:]', Matrix{Float32}(reshape(testTargets[1,:], 1, :))), maxEpochs=modelHyperparameters["maxEpochs"], learningRate=modelHyperparameters["learningRate"]);
                        testInputsMatrix = Matrix{Float32}(testInputs[1, :]')  # Convertir las entradas a tipo Float32
                        testTargetsMatrix = Matrix{Bool}(reshape(testTargets[1, :], 1, :))
                        ann, = trainClassANN(modelHyperparameters["topology"],
                            (trainingInputs', trainingTargets'),
                            validationDataset=(validationInputs', validationTargets'),
                            testDataset=(testInputsMatrix', testTargetsMatrix'),
                            maxEpochs=modelHyperparameters["maxEpochs"],
                            learningRate=modelHyperparameters["learningRate"],
                            maxEpochsVal=modelHyperparameters["maxEpochsVal"]);
                    else
                        # Si no se desea usar conjunto de validacion, se entrena unicamente con conjuntos de entrenamiento y test,
                        # teniendo cuidado de codificar las salidas deseadas correctamente
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
                    testOutPut = testOutPut .> 0.5
                    
                    testTargets = testTargets .> 0.5
                    testOutPut = [testOutPut[i] for i in 1:length(testOutPut)];
                    testTargets2 = [testTargets[i] for i in 1:length(testTargets)];
                    # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
                    (testAccuraciesEachRepetition[numTraining], _, _, _, _, _, testF1EachRepetition[numTraining], _) = confusionMatrix(vec(testOutPut), vec(testTargets2));
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