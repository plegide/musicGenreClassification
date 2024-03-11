using WAV

function dividir_audio(input_file::AbstractString, output_folder::AbstractString, segment_length::Int, overlap::Int)
    # Cargar el archivo de audio
    audio, sample_rate = wavread(input_file)

    # Calcular el número total de muestras y el número de segmentos necesarios
    total_samples = length(audio)
    num_segments = ceil(Int, total_samples / (segment_length - overlap))

    # Iterar sobre los segmentos y extraerlos
    for i in 1:num_segments
        # Calcular los índices de inicio y fin para cada segmento
        start_index = (i - 1) * (segment_length - overlap) + 1
        end_index = min(start_index + segment_length - 1, total_samples)

        # Asegurarse de que el segmento tiene la longitud deseada
        segment = zeros(segment_length)
        segment[1:end_index - start_index + 1] = audio[start_index:end_index]

        # Guardar el segmento como un nuevo archivo
        output_file = joinpath(output_folder, "segmento_$i.wav")
        wavwrite(segment, output_file, Fs=sample_rate)
    end
end

# Ejemplo de uso
input_file = "audio.wav"
output_folder = "segmentos"
segment_length = 65536  # Longitud de cada segmento en muestras
overlap = 16384  # Solapamiento en muestras

dividir_audio(input_file, output_folder, segment_length, overlap)
