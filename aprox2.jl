using ScikitLearn, Flux, Flux.Losses, DelimitedFiles, Plots, ProgressMeter, Random, Statistics
#push!(LOAD_PATH, "fonts")
include("fonts/funciones.jl")
@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier

# Fijamos la semilla aleatoria para poder repetir los experimentos
Random.seed!(1);

numFolds = 10;

# Parametros principales de la RNA y del proceso de entrenamiento
topology = [5]; # Dos capas ocultas con 4 neuronas la primera y 3 la segunda
learningRate = 0.01; # Tasa de aprendizaje
maxEpochs = 1000; # Numero maximo de ciclos de entrenamiento
validationRatio = 0.2; # Porcentaje de patrones que se usaran para validacion. Puede ser 0, para no usar validacion
maxEpochsVal = 20; # Numero de ciclos en los que si no se mejora el loss en el conjunto de validacion, se para el entrenamiento
numRepetitionsAANTraining = 50; # Numero de veces que se va a entrenar la RNA para cada fold por el hecho de ser no determinístico el entrenamiento

# Parametros del SVM
kernel = "rbf";
kernelDegree = 3;
kernelGamma = 2;
C=1;

# Parametros del arbol de decision
maxDepth = 1;

# Parapetros de kNN
numNeighbors = 3;

# Cargamos el dataset
dataset = readdlm("datasets/aprox2/aprox2.data",',');
inputs = dataset[:,1:3];
# Preparamos las entradas y las salidas deseadas
inputs = convert(Array{Float32,2}, inputs);
targets = dataset[:,4];

# Normalizamos las entradas, a pesar de que algunas se vayan a utilizar para test

normalizeMinMax!(inputs);

# Entrenamos las RR.NN.AA.
modelHyperparameters = Dict();
modelHyperparameters["topology"] = topology;
modelHyperparameters["learningRate"] = learningRate;
modelHyperparameters["validationRatio"] = validationRatio;
modelHyperparameters["numExecutions"] = numRepetitionsAANTraining;
modelHyperparameters["maxEpochs"] = maxEpochs;
modelHyperparameters["maxEpochsVal"] = maxEpochsVal;
modelCrossValidation(:ANN, modelHyperparameters, inputs, targets, numFolds);


# # Entrenamos las SVM
# modelHyperparameters = Dict();
# modelHyperparameters["kernel"] = kernel;
# modelHyperparameters["kernelDegree"] = kernelDegree;
# modelHyperparameters["kernelGamma"] = kernelGamma;
# modelHyperparameters["C"] = C;
# modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, numFolds);

# # Entrenamos los arboles de decision
# modelCrossValidation(:DecisionTree, Dict("maxDepth" => maxDepth), inputs, 
# targets, numFolds);

# Entrenamos los kNN
# modelCrossValidation(:kNN, Dict("numNeighbors" => numNeighbors), inputs, 
# targets, numFolds);

