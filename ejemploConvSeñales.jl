using Flux
using Flux.Losses
using Flux: onehotbatch, onecold
using JLD2, FileIO
using Statistics: mean


##############################################################################################################################################################################
#
# Codigo similar a "ejemploConv.jl"
#

train_imgs   = load("MNIST.jld2", "train_imgs");
labels = 0:9; # Las etiquetas

function convertirArrayImagenesHWCN(imagenes)
    numPatrones = length(imagenes);
    nuevoArray = Array{Float32,4}(undef, 28, 28, 1, numPatrones); # Importante que sea un array de Float32
    for i in 1:numPatrones
        @assert (size(imagenes[i])==(28,28)) "Las imagenes no tienen tamaño 28x28";
        nuevoArray[:,:,1,i] .= imagenes[i];
    end;
    return nuevoArray;
end;
train_imgs = convertirArrayImagenesHWCN(train_imgs);


funcionTransferenciaCapasConvolucionales = relu;


##############################################################################################################################################################################
#
# Pruebas para crear una red convolucional para clasificación de señales
#

# Tomo sólo las 100 primeras para que estas pruebas vayan más rápido
N = 100;

# Construimos la matriz tridimensional con las señales, donde cada dimensión es:
#  1 - La señal
#  2 - Canal (en este ejemplo sólo hay un canal, pero podría haber varias señales)
#  3 - El número de instancia
inputs = Array{Float32,3}(undef, 28*28, 1, N);
for i in 1:N
    inputs[:,1,i] .= reshape(train_imgs[:,:,1,i], 28*28);
end;
println("Tamaño de la matriz de entrenamiento: ", size(inputs))
println("   Longitud de la señal: ", size(inputs,1))
println("   Numero de canales: ", size(inputs,2))
println("   Numero de instancias: ", size(inputs,3))

ann = Chain(
    Conv((3,), 1=>16, pad=1, funcionTransferenciaCapasConvolucionales),
    MaxPool((2,)),
    Conv((3,), 16=>32, pad=1, funcionTransferenciaCapasConvolucionales),
    MaxPool((2,)),
    Conv((3,), 32=>32, pad=1, funcionTransferenciaCapasConvolucionales),
    MaxPool((2,)),
    x -> reshape(x, :, size(x, 3)),
    Dense(3136, 10),
    softmax
);

println("Tamaño de las salidas: ", size(ann(inputs)));