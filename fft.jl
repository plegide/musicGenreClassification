using FFTW
using Statistics
using WAV
using FileIO

function audioFft(audio_file_path::AbstractString, genre::AbstractString)
    audio_data, Fs = wavread(audio_file_path)

    n = length(audio_data)
    senalFrecuencia = abs.(fft(audio_data));

    if (iseven(n))
        @assert(mean(abs.(senalFrecuencia[2:Int(n/2)] .- senalFrecuencia[end:-1:(Int(n/2)+2)]))<1e-8);
        senalFrecuencia = senalFrecuencia[1:(Int(n/2)+1)];
    else
        @assert(mean(abs.(senalFrecuencia[2:Int((n+1)/2)] .- senalFrecuencia[end:-1:(Int((n-1)/2)+2)]))<1e-8);
        senalFrecuencia = senalFrecuencia[1:(Int((n+1)/2))];
    end;

    file_path = "genres.data"
    if !isfile(file_path)
        touch(file_path)
    end
    open(file_path, "a") do file
        write(file, "$(mean(senalFrecuencia)),$(std(senalFrecuencia)),$(genre)\n")
    end
end;

println("[!] Processing audio files")

genres = readdir("segments")

for genre in genres
    if(isdir(joinpath("segments", genre)) == false)
        continue
    end
    for audio in readdir(joinpath("segments", genre))
        if endswith(audio, ".wav")
            try
                audioFft(joinpath("segments", genre, audio), genre)
            catch
                println("[x] Error processing $(audio)")
            end
        end
    end
end

println("[!] Finished processing audio files")
println("[!] Generated genres.data file with audio features")