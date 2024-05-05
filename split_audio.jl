using WAV

function dividir_audio(input_file::AbstractString, output_folder::AbstractString, segment_length::Int, overlap::Int)
    # Lee el audio
    audio, sample_rate = wavread(input_file)

    total_samples = length(audio)
    num_segments = ceil(Int, total_samples / (segment_length - overlap))
    for i in 1:num_segments-1
        # Calcula el principio y el fin del segmento en concreto (hay overlap entre segmentos)
        start_index = (i - 1) * (segment_length - overlap) + 1
        end_index = min(start_index + segment_length - 1, total_samples)
        
        # Copia el segmento del audio 
        segment = zeros(segment_length)
        segment[1:end_index - start_index + 1] = audio[start_index:end_index]

        if !isdir(output_folder)
            mkdir(output_folder)
        end
        output_file = joinpath(output_folder, "$(basename(input_file))_segment_$(i).wav")
        # Guarda el segmento como un wav
        wavwrite(segment, output_file, Fs=sample_rate)
    end
        # El mismo proceso pero hecho para el ultimo segmento (hay mas overlap entre el penultimo segmento y el ultimo que entre el resto)
        # start_index = (total_samples - segment_length) + 1
        # end_index = total_samples
        # segment = zeros(segment_length)
        # segment[1:end_index - start_index + 1] = audio[start_index:end_index]
        # if !isdir(output_folder)
        #     mkdir(output_folder)
        # end
        # output_file = joinpath(output_folder, "$(basename(input_file))_segment_$(num_segments).wav")
        # wavwrite(segment, output_file, Fs=sample_rate)
end

println("[!] Processing audio files")

# Recorre le directorio en donde estan guardados los trozos de audio originales (los de la BD),
# los cuales estan en carpetas con el nombre de su genero. Cada uno de estos es pasado a la funcion 
# dividir_audio
for genre in readdir("genres")
    if(isdir(joinpath("genres", genre)) == false)
        continue
    end
    println("Procesando $(genre)...")
    for audio in readdir(joinpath("genres", genre))
        if endswith(audio, ".wav")
            try
                dividir_audio(joinpath("genres", genre, audio), joinpath("segments", genre), 65536, 16384)
            catch
                println("[x] Error processing $(audio)")
            end
        end
    end
end

println("[!] Finished processing audio files")