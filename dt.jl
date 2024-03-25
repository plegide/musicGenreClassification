using ScikitLearn, DelimitedFiles, Random

# Carga de ScikitLearn
@sk_import tree: DecisionTreeClassifier

# Genera el modelo
model = DecisionTreeClassifier(max_depth=4, random_state=1) 

function holdOut(N::Int, P::Float64)
    @assert ((P>=0.) & (P<=1.));
    indices = randperm(N);
    numTrainingInstances = Int(round(N*(1-P)));
    return (indices[1:numTrainingInstances], indices[numTrainingInstances+1:end]);
end


# 1. Cargar el conjunto de datos
dataset = readdlm("classicOrMetal.data",',');
inputs = dataset[:,1:2];
targets = dataset[:,3];

(trainIndexes, testIndexes) = holdOut(size(inputs,1), 0.20);

trainingInputs = inputs[trainIndexes,:];
testInputs = inputs[testIndexes,:];
trainingTargets = targets[trainIndexes,:];
testTargets = targets[testIndexes,:];


# Entrenar el modelo
fit!(model, trainingInputs, trainingTargets); 

# Hacer prediciones 
predictions = predict(model, testInputs)

accuracy = sum(testTargets .== predictions) / length(testTargets)
println("Exactitud: ", accuracy)

