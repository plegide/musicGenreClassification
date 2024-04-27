using FFTW
using Statistics
using WAV
using FileIO
using MFCC
using DSP
using AcousticFeatures

function audioFft(audio_file_path::AbstractString)
    # Lee el archivo de audio
    wav_data, Fs = wavread(audio_file_path)

    n = length(wav_data)
    senalFrecuencia = abs.(fft(wav_data));

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
    wav_data = vec(wav_data)

    # Square the values
    squared_values = wav_data .^ 2

    # Compute the mean of the squared values
    mean_squared = mean(squared_values)

    # Take the square root of the mean squared value
    rms = sqrt(mean_squared)

    return rms
end


function compute_mfcc(filename::String)
    wav_data, sample_rate = wavread(filename)
    mfccs, _, _ = mfcc(wav_data, sample_rate)
    mean_mfccs = mean(mfccs)
    std_mfccs = std(mfccs)
    return mean_mfccs, std_mfccs
end

function compute_zero_crossing_rate(filename::String)
    wav_data, _ = wavread(filename)
    zero_crossings = 0
    # Calculate zero crossings
    for i in 1:length(wav_data)-1
        if (wav_data[i] > 0 && wav_data[i+1] < 0) || (wav_data[i] < 0 && wav_data[i+1] > 0)
            zero_crossings += 1
        end
    end
    # Calculate zero crossing rate
    zero_crossing_rate = zero_crossings / (length(wav_data) - 1)
    return zero_crossing_rate

end


# function compute_stft(audio, window_size, hop_size)
#     # Calculate number of frames
#     num_frames = 1 + div(size(audio, 1) - window_size, hop_size)
    
#     # Pre-allocate STFT matrix
#     stft = zeros(Complex{Float64}, window_size ÷ 2 + 1, num_frames)
    
#     # Compute STFT
#     for i in 1:num_frames
#         frame_start = (i - 1) * hop_size + 1
#         frame_end = frame_start + window_size - 1
#         frame = audio[frame_start:frame_end, 1] .* hamming(window_size)
#         stft[:, i] = fft(frame, window_size)[1:window_size ÷ 2 + 1]
#     end
    
#     return stft
# end

# Compute spectral centroid
function compute_spectral_centroid(filename::String)
    wav_data, sample_rate = wavread(filename)
    x = Score(SpectralCentroid(), wav_data; fs=sample_rate)
    return x[1]

end

function compute_spectral_flatness(filename::String)
    wav_data, sample_rate = wavread(filename)
    x = Score(SpectralFlatness(), wav_data; fs=sample_rate)
    return x[1]

end

function compute_myriad(filename::String)
    wav_data, sample_rate = wavread(filename)
    x = Score(Myriad(), wav_data; fs=sample_rate)
    return x[1]
end

function compute_permutation_entropy(filename::String)
    wav_data, sample_rate = wavread(filename)
    x = Score(PermutationEntropy(5, 1, true, true), wav_data; fs=sample_rate)
    return x[1]
end

function compute_energy(filename::String)
    wav_data, sample_rate = wavread(filename)
    x = Score(Energy(), wav_data; fs=sample_rate)
    return x[1]
end

function compute_sound_pressure(filename::String)
    wav_data, sample_rate = wavread(filename)
    x = Score(SoundPressureLevel(), wav_data; fs=sample_rate)
    return x[1]
end




println("[!] Processing audio files")

genres = readdir("segments")

# Itera sobre los subdirectorios que hay dentro de segments (uno por cada genero) y llama a la function 
# audioFft sobre cada uno de los segmentos de audio
file_path = "aprox6.data"
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
                meanMFCC, stdMFCC = compute_mfcc(joinpath("segments", genre, audio))
                zeroCrossRate = compute_zero_crossing_rate(joinpath("segments", genre, audio))
                spectralCentroid = compute_spectral_centroid(joinpath("segments", genre, audio))
                spectralFlatness = compute_spectral_flatness(joinpath("segments", genre, audio))
                myriad = compute_myriad(joinpath("segments", genre, audio))
                permutation_entropy = compute_permutation_entropy(joinpath("segments", genre, audio))
                if(isequal(meanSF, NaN) || isequal(stdSF, NaN) || isequal(rms, NaN) || isequal(meanMFCC, NaN) || isequal(stdMFCC, NaN) || isequal(zeroCrossRate, NaN) || isequal(spectralCentroid, NaN) || isequal(spectralFlatness, NaN) || isequal(permutation_entropy, NaN) || isequal(myriad, NaN))
                    println("[x] Error processing $(audio)")
                    continue
                end
                open(file_path, "a") do file
                    write(file, "$(meanSF),$(stdSF),$(rms),$(meanMFCC),$(stdMFCC),$(zeroCrossRate),$(spectralCentroid),$(spectralFlatness),$(myriad),$(permutation_entropy),$(genre)\n")
                end
            catch
                println("[x] Error processing $(audio)")
            end
        end
    end
end

println("[!] Finished processing audio files")
println("[!] Generated $(file_path) file with audio features")