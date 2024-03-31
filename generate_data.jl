using FFTW
using Statistics
using WAV
using FileIO

function audioFft(audio_file_path::AbstractString)
    # Lee el archivo de audio
    audio_data, Fs = wavread(audio_file_path)

    n = length(audio_data)
    senalFrecuencia = abs.(fft(audio_data));

    # Coje solo la primera parte de la senal debido a que las dos mitades son simetricas
    if (iseven(n))
        @assert(mean(abs.(senalFrecuencia[2:Int(n/2)] .- senalFrecuencia[end:-1:(Int(n/2)+2)]))<1e-8);
        senalFrecuencia = senalFrecuencia[1:(Int(n/2)+1)];
    else
        @assert(mean(abs.(senalFrecuencia[2:Int((n+1)/2)] .- senalFrecuencia[end:-1:(Int((n-1)/2)+2)]))<1e-8);
        senalFrecuencia = senalFrecuencia[1:(Int((n+1)/2))];
    end;
    return mean(senalFrecuencia), std(senalFrecuencia)
end;

function compute_rms(filename::String)
    # Load the WAV file
    wav_data, sample_rate = wavread(filename)

    # Extract the audio data from the WAV file
    audio_data = vec(wav_data)

    # Square the values
    squared_values = audio_data .^ 2

    # Compute the mean of the squared values
    mean_squared = mean(squared_values)

    # Take the square root of the mean squared value
    rms = sqrt(mean_squared)

    return rms
end

println("[!] Processing audio files")

genres = readdir("segments")

# Itera sobre los subdirectorios que hay dentro de segments (uno por cada genero) y llama a la function 
# audioFft sobre cada uno de los segmentos de audio
file_path = "aprox2.data"
if !isfile(file_path)
    touch(file_path)
end

for genre in genres
    if(isdir(joinpath("segments", genre)) == false)
        continue
    end
    for audio in readdir(joinpath("segments", genre))
        if endswith(audio, ".wav")
            try
                meanSF, stdSF = audioFft(joinpath("segments", genre, audio))
                rms = compute_rms(joinpath("segments", genre, audio))
                open(file_path, "a") do file
                    write(file, "$(meanSF),$(stdSF),$(rms),$(genre)\n")
                end
            catch
                println("[x] Error processing $(audio)")
            end
        end
    end
end

println("[!] Finished processing audio files")
println("[!] Generated $(file_path) file with audio features")