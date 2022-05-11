#Importing library and thir function
import pyaudio
import wave
from pydub import AudioSegment
from pydub.silence import split_on_silence
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
record()
#reading from audio mp3 file
sound = AudioSegment.from_mp3("recordedFile.wav")
# spliting audio files
audio_chunks = split_on_silence(sound, min_silence_len=200, silence_thresh=-40 )
list_of_chunk_files=[]
#loop is used to iterate over the output list
for i, chunk in enumerate(audio_chunks):
   output_file = "chunk{0}.wav".format(i)
   list_of_chunk_files.append("chunk{0}.wav".format(i))
   print("Exporting file", output_file)
   chunk.export(output_file, format="wav")
# chunk files saved as Output  


def writing_file(predictions):
            
            # Program to show various ways to read and
            # write data in a file.
            file1 = open("Notes.txt","w")

            file1.writelines(predictions)
            file1.close() #to change file access modes
            
writing_file(predictions)