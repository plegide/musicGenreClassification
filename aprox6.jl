using ScikitLearn, Flux, Flux.Losses, DelimitedFiles, Plots, ProgressMeter, Random, Statistics
#push!(LOAD_PATH, "fonts")
include("fonts/funciones.jl")
@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier

Random.seed!(1);


# Cargamos el dataset
# JUAN CURRA


# Entrenamos deepLearning
modelHyperparameters = Dict();
# modelHyperparameters["topology"] = topology;


deepLearning(modelHyperparameters);


