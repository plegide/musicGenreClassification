using WAV

function split_audio(filename, segment_length, overlap)
    # Read the audio file
    y, fs = wavread(filename)
    # Get the number of samples and the segment and overlap in samples
    total_samples = length(y)
    segment_samples = round(Int, segment_length * fs)
    overlap_samples = round(Int, overlap * fs)
    start_sample = 1
    

    i = 1
    while start_sample < total_samples
        end_sample = min(start_sample + segment_samples - 1, total_samples)
        segment = y[start_sample:end_sample, :]
        wavwrite(segment, "segment_$i.wav", Fs=fs)
        start_sample += segment_samples - overlap_samples
        i += 1
    end
end

# Call the function with the filename, segment length in seconds, and overlap in seconds
split_audio("audio.wav", 2.97, 0.3)