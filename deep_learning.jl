using ScikitLearn, Flux, Flux.Losses, DelimitedFiles, Plots, ProgressMeter, Random, Statistics
#push!(LOAD_PATH, "fonts")
include("fonts/funciones.jl")

Random.seed!(1);


# Cargamos el dataset


# Entrenamos deepLearning
modelHyperparameters = Dict();
# modelHyperparameters["topology"] = topology;


deepLearning(modelHyperparameters);


