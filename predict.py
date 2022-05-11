#!pip install pydub
from pydub import AudioSegment
from pydub.silence import split_on_silence
import pyaudio
import wave
from keras.models import load_model
#preprocessing
# An integer scalar Tensor. The window length in samples.
frame_length = 256
# An integer scalar Tensor. The number of samples to step.
frame_step = 160
# An integer scalar Tensor. The size of the FFT to apply.
# If not provided, uses the smallest power of 2 enclosing frame_length.
fft_length = 384
# The set of characters accepted in the transcription.
characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
# Mapping characters to integers
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
# Mapping integers back to original characters
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)
def record():

            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 16000
            CHUNK = 512
            RECORD_SECONDS = 10
            WAVE_OUTPUT_FILENAME = "recordedFile.wav"
            device_index = 2
            audio = pyaudio.PyAudio()

            print("----------------------record device list---------------------")
            info = audio.get_host_api_info_by_index(0)
            numdevices = info.get('deviceCount')
            for i in range(0, numdevices):
                    if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                        print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))

            print("-------------------------------------------------------------")

            index = int(input())
            print("recording via index "+str(index))

            stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,input_device_index = index,
                            frames_per_buffer=CHUNK)
            print ("recording started")
            Recordframes = []
            
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                Recordframes.append(data)
            print ("recording stopped")
            
            stream.stop_stream()
            stream.close()
            audio.terminate()

            waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            waveFile.setnchannels(CHANNELS)
            waveFile.setsampwidth(audio.get_sample_size(FORMAT))
            waveFile.setframerate(RATE)
            waveFile.writeframes(b''.join(Recordframes))
            waveFile.close()


def chunk_audio(file_name):
    #reading from audio mp3 file
    sound = AudioSegment.from_mp3(file_name)
    # spliting audio files
    audio_chunks = split_on_silence(sound, min_silence_len=200, silence_thresh=-40 )
    list_of_chunk_files=[]
    #loop is used to iterate over the output list
    for i, chunk in enumerate(audio_chunks):
      output_file = "chunk{0}.wav".format(i)
      list_of_chunk_files.append("/content/chunk{0}.wav".format(i))
      print("Exporting file", output_file)
      chunk.export(output_file, format="wav")
    return list_of_chunk_files

def writing_file(predictions):
            predictions=" ".join(predictions)
            # Program to show various ways to read and
            # write data in a file.
            file1 = open("Notes.txt","w")

            file1.writelines(predictions)
            file1.close() #to change file access modes
       
def pre_single_sample(wav_file):
    ###########################################
    ##  Process the Audio
    ##########################################
    # 1. Read wav file
    file = tf.io.read_file(wav_file)
    # 2. Decode the wav file
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    # 3. Change type to float
    audio = tf.cast(audio, tf.float32)
    # 4. Get the spectrogram
    spectrogram = tf.signal.stft(
        audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )
    # 5. We only need the magnitude, which can be derived by applying tf.abs
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    # 6. normalisation
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)
 
    return spectrogram
# A utility function to decode the output of the network

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text

def predictions(file_list,model):
        predictions = []
        pred_data = tf.data.Dataset.from_tensor_slices(
            (file_list)
        )
        pred_dataset = (
          pred_data.map(pre_single_sample,num_parallel_calls=tf.data.AUTOTUNE)
            .padded_batch(1)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        for batch in pred_dataset:  
            X = batch
            batch_predictions = model.predict(X)
            batch_predictions = decode_batch_predictions(batch_predictions)
            predictions.extend(batch_predictions)
        return predictions            
model=load_model('/model/my_model.h5',custom_objects={'CTCLoss': CTCLoss})
record()
fil=chunk_audio('recordedFile.wav')
predictions= predictions(fil,model)
writing_file(predictions)
