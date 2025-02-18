import sounddevice as sd
import numpy as np
import threading, queue
import warnings, time

# file handlers 
import os, shutil, wave, tempfile
from scipy.io.wavfile import write

# audio conversion class to handle different types of audio inputs
import librosa

class AudioConverter:
    def __init__(self, target_samplerate: int, target_channels: int, target_dtype: type = np.int16, chunk_size: int = 1024):
        """
        Initialize the AudioConverter.

        Args:
            target_samplerate (int): Desired sample rate.
            target_channels (int): Desired number of channels.
            target_dtype (type): Desired output data type (e.g., np.int16, np.float32).
            chunk_size (int): Number of frames to process at a time (default: 1024).
        """
        self.target_samplerate = target_samplerate
        self.target_channels = target_channels
        self.target_dtype = target_dtype
        self.chunk_size = chunk_size

    def load_and_convert_wav(self, filename: str):
        """
        Load a WAV file in chunks and convert it to the target sample rate, number of channels, and data type.

        Args:
            filename (str): Path to the WAV file.

        Yields:
            np.ndarray: Converted audio chunks in the target data type (shape: [chunk_size, target_channels]).
        """
        with wave.open(filename, 'rb') as wf:
            input_samplerate = wf.getframerate()
            input_channels = wf.getnchannels()

            # Buffer to store leftover samples between chunks
            buffer = np.zeros((0, input_channels), dtype=np.float32)

            while True:
                # Read a chunk of raw audio data
                raw_chunk = wf.readframes(self.chunk_size)
                if not raw_chunk:
                    break

                # Convert raw data to numpy array and normalize to float32
                chunk = np.frombuffer(raw_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                chunk = chunk.reshape(-1, input_channels)

                # Append the chunk to the buffer
                buffer = np.vstack((buffer, chunk))

                # Resample the buffer if necessary (using librosa for high-quality resampling)
                if input_samplerate != self.target_samplerate:
                    buffer = self.resample_audio(buffer, input_samplerate, self.target_samplerate)

                # Convert channels if necessary
                if input_channels != self.target_channels:
                    buffer = self.convert_channels(buffer, input_channels, self.target_channels)

                # Scale and convert to the target data type
                while buffer.shape[0] >= self.chunk_size:
                    chunk = buffer[:self.chunk_size, :]
                    chunk = self.scale_to_target_dtype(chunk)  # Scale and convert to target data type
                    yield chunk
                    buffer = buffer[self.chunk_size:, :]

            # Yield any remaining samples in the buffer
            if buffer.shape[0] > 0:
                buffer = self.scale_to_target_dtype(buffer)  # Scale and convert to target data type
                yield buffer

    def resample_audio(self, data: np.ndarray, input_samplerate: int, target_samplerate: int) -> np.ndarray:
        """
        Resample audio data using librosa for high-quality resampling.

        Args:
            data (np.ndarray): Input audio data (shape: [samples, channels]).
            input_samplerate (int): Sample rate of the input audio.
            target_samplerate (int): Desired sample rate.

        Returns:
            np.ndarray: Resampled audio data in float32 format.
        """
        if input_samplerate == target_samplerate:
            return data

        # Resample each channel separately using librosa
        resampled_data = np.zeros((0, data.shape[1]), dtype=np.float32)
        for channel in range(data.shape[1]):
            channel_data = data[:, channel]
            resampled_channel = librosa.resample(
                channel_data, orig_sr=input_samplerate, target_sr=target_samplerate
            )
            resampled_data = np.vstack((resampled_data, resampled_channel.reshape(-1, 1)))

        return resampled_data

    def convert_channels(self, data: np.ndarray, input_channels: int, target_channels: int) -> np.ndarray:
        """
        Convert audio data to the target number of channels.

        Args:
            data (np.ndarray): Input audio data (shape: [samples, input_channels]).
            input_channels (int): Number of channels in the input audio.
            target_channels (int): Desired number of channels.

        Returns:
            np.ndarray: Converted audio data in float32 format.
        """
        if input_channels == target_channels:
            return data

        if input_channels == 1 and target_channels == 2:
            # Mono to stereo: duplicate the single channel
            return np.repeat(data, target_channels, axis=1)
        elif input_channels == 2 and target_channels == 1:
            # Stereo to mono: use librosa.to_mono for balanced conversion
            return librosa.to_mono(data.T).reshape(-1, 1)
        else:
            raise ValueError(f"Unsupported channel conversion: {input_channels} -> {target_channels}")

    def scale_to_target_dtype(self, data: np.ndarray) -> np.ndarray:
        """
        Scale and convert audio data to the target data type.

        Args:
            data (np.ndarray): Input audio data in float32 format.

        Returns:
            np.ndarray: Scaled and converted audio data in the target data type.
        """
        if self.target_dtype == np.int16:
            # Normalize to avoid clipping and scale to int16 range
            data = librosa.util.normalize(data, axis=0)  # Normalize to [-1, 1]
            return (data * 32767).astype(np.int16)
        elif self.target_dtype == np.float32:
            # No scaling needed for float32
            return data.astype(np.float32)
        else:
            raise ValueError(f"Unsupported target data type: {self.target_dtype}")


# example use

# converter = AudioConverter(target_samplerate=44100, target_channels=2, target_dtype=np.float32)
# for chunk in converter.load_and_convert_wav("dataset/speaker2/odd1out_s2.wav"):
#     print(chunk.dtype)  # Output: float32




"""

EOF Error: This error occurs when the input file is empty

"""


class audio_handler:

    class record:
        def __init__(self):

            #================================================================
            #                    recording variables
            #================================================================

            self.sample_rate = self.detect_sample_rate()
            self.channels = len([_ for _ in sd.default.channels])
            self.depth = 0

            self.recording = False
            self.stream = None

            self.audio_data = []
            self.audio_data_lock = threading.Lock()
            self.volume = 1.0  # Default volume multiplier
            self.save_period = 100  # Save audio data after this many frames have been recorded
            self.save_counter = 0
            
            self.TEMP_FOLDER = tempfile.mkdtemp()
            self.number_of_chunks = 0
            self.threads = []

            self.saving = False
            self.instant_stop = False

        def detect_sample_rate(self)-> int:

            """
            Detects the sample rate of the default input device.
            """

            try:
                # Get the default input device info
                default_input = sd.query_devices(kind='input', device=None)

                # Extract the sample rate from the device info
                sample_rate = int(default_input['default_samplerate'])
                
                return sample_rate

            except Exception as e:
                warnings.warn(f"Error detecting sample rate: {e}")
                return 44100 # lowest typical sample rate of 44.1kh

        def set_volume(self, volume):
            """Set the volume multiplier."""
            if volume < 0:
                warnings.warn("Volume must be non-negative!")
                return
            self.volume = volume

        def _callback(self, indata, frames, time, status):
            if status:
                warnings.warn(f"_callback Status: {status}")
            with self.audio_data_lock:
                self.audio_data.append(indata.copy())
            
            #save audio data after 10 chunks have been recorded
            if self.save_counter > self.save_period: 
                self._save_audio_chunk()
                self.save_counter = 0

            self.save_counter += 1

            print(" "*15, end="\r")
            print(f"Recording: {self.save_counter}", end= "\r")

        def start_recording(self):
            if self.recording:
                warnings.warn("Already recording!")
                return

            self.recording = True
            self.audio_data = []
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self._callback
            )
            self.stream.start()

        def _temp_folder_cleanup(self):
            if not os.path.exists(self.TEMP_FOLDER):
                    warnings.warn("Temp folder does not exist or was not created")

            # Remove all temporary audio chunks from the temp folder.
            shutil.rmtree(self.TEMP_FOLDER)

        def stop_recording(self):
            if not self.recording:
                return

            self.recording = False
            self.stream.stop()
            self.stream.close()
            
            while self.saving and not self.instant_stop:
                time.sleep(0.1)
                print("waiting for saving to finish...", end="\r")
            self._temp_folder_cleanup()

        def save_recording(self, filename):
            """Save the stored audio to a WAV file."""

            self.saving = True
            sample_width = None

            file = os.listdir(self.TEMP_FOLDER)[0]
            file_path = os.path.join(self.TEMP_FOLDER, file)

            with wave.open(file_path, "r") as f:
                sample_width = f.getsampwidth()

            # Open the output file for writing.
            def _concatenate_wav_files(num_channels, sample_rate, sample_width, input_files, output_file, max_files):
                """
                Concatenates multiple WAV files into a single output file without loading them into memory.

                Args:
                    num_channels: Number of audio channels (e.g., 1 for mono, 2 for stereo).
                    sample_rate: Sample rate of the audio (e.g., 44100 Hz).
                    sample_width: Sample width in bytes (e.g., 2 for 16-bit audio).
                    input_files: A list of paths to the input WAV files.
                    output_file: The path to the output WAV file.
                    max_files: Maximum number of files to concatenate.
                """

                try:
                    # Open the output file for writing.
                    with wave.open(output_file, 'wb') as wf:
                        wf.setnchannels(num_channels)
                        wf.setsampwidth(sample_width)
                        wf.setframerate(sample_rate)

                        num_files_appended = 0

                        # Iterate over the input files and append their audio data to the output file.
                        for input_file in input_files:
                            if num_files_appended >= max_files:
                                warnings.warn(f"Reached maximum number of files ({max_files}) to concatenate.")
                                break

                            try:
                                with wave.open(input_file, 'rb') as infile:
                                    # Verify that the input file has the same parameters as the output file.
                                    if (infile.getnchannels() != num_channels or
                                        infile.getsampwidth() != sample_width or
                                        infile.getframerate() != sample_rate):
                                        warnings.warn(f"Skipping {input_file}: mismatched audio parameters.")
                                        continue

                                    # Read the audio data in chunks and write it to the output file.
                                    while True:
                                        chunk = infile.readframes(1024)
                                        if not chunk:
                                            break
                                        wf.writeframes(chunk)
                                    
                                    

                                num_files_appended += 1
                            except Exception as e:
                                warnings.warn(f"Error processing {input_file}: {e}")
                                continue
                    

                except Exception as e:
                    warnings.warn(f"Error during concatenation: {e}")
                
                finally:
                    self.saving = False


            thread = threading.Thread(target=_concatenate_wav_files, 
                                args=(self.channels, 
                                self.sample_rate, 
                                sample_width, 
                                [self.TEMP_FOLDER + "/" + _ for _ in os.listdir(self.TEMP_FOLDER)], 
                                filename, 
                                self.number_of_chunks         )) 
            thread.start()
            self.threads.append(thread)

        def _save_audio_chunk(self):
            """Save the recorded audio to a WAV file as int16."""
            if not self.audio_data:
                warnings.warn("No audio data to save! Note: this is from the _save_audio_chunk method.")
                return

            with self.audio_data_lock:
                # Concatenate audio data and apply volume
                audio_buffer = np.concatenate(self.audio_data.copy(), axis=0)
                self.audio_data = []  # Clear audio data for next chunk
                audio_buffer = np.clip(audio_buffer * self.volume, -1.0, 1.0)  # Apply volume multiplier

                # Scale to int16 range
                audio_buffer_int16 = np.int16(audio_buffer * 32767)  

            # Save into temp folder
            file_path = os.path.join(self.TEMP_FOLDER, f"AC_{self.number_of_chunks}.wav")
            write(file_path, self.sample_rate, audio_buffer_int16)

            #increment chunk number to preserve order without having them listed 
            self.number_of_chunks += 1

    class play:
        def __init__(self, channels: int = 2, blocksize: int = 1024):


            OUTPUT_FORMAT = self.get_output_format()
            self.samplerate = OUTPUT_FORMAT["samplerate"]
            self.channels = channels
            # len([_ for _ in sd.default.channels])
            self.blocksize = blocksize
            self.sample_width = OUTPUT_FORMAT["sample_width"]

            self.play_audio_queue = queue.Queue()
            self.file_queue = queue.Queue()
            self.loading_thread = False

            self.play_queue_lock = threading.Lock()
            self.play_stream = None

            self.playback_thread = None
            self.stop_playing_audio_event = threading.Event()
            self.underrun_count = 0

        def get_output_format(self, device_id: int = None) -> dict:
            """
            Get the output format (frequency, sample width, and data type) of the specified audio device.

            Args:
                device_id (int, optional): The ID of the audio device. If None, the default output device is used.

            Returns:
                dict: A dictionary containing the output format details:
                    - 'frequency': The default sample rate in Hz.
                    - 'width': The sample width in bytes.
                    - 'dtype': The numpy data type (e.g., np.int16, np.float32).
            """
            # Get device info
            device_info = sd.query_devices(device_id, 'output')

            # Extract frequency (sample rate)
            frequency = int(device_info['default_samplerate'])

            # Extract sample width (in bytes)
            # Note: This is not directly provided by sounddevice, so we infer it from the default format
            default_format = device_info.get('default_speaker_format', 'int16')
            if default_format == 'int16':
                width = 2  # 16 bits = 2 bytes
                dtype = np.int16
            elif default_format == 'int32':
                width = 4  # 32 bits = 4 bytes
                dtype = np.int32
            elif default_format == 'float32':
                width = 4  # 32 bits = 4 bytes
                dtype = np.float32
            elif default_format == 'float64':
                width = 8  # 64 bits = 8 bytes
                dtype = np.float64
            else:
                # Default to int16 if format is unknown
                width = 2
                dtype = np.int16

            return {
                'samplerate': frequency,
                'sample_width': width,
                'dtype': dtype,
                'channels': 2,
            }


        def _load_wav_chunks(self, filename: str):
            """Thread-safe WAV file loader with chunked reading and auto conversion"""

            converter = AudioConverter(target_samplerate=48000, target_channels=2, target_dtype=np.float32)
            try:
                for chunk in converter.load_and_convert_wav(filename):
                    with self.play_queue_lock:
                        self.play_audio_queue.put(chunk)
            
            except Exception as e:
                warnings.warn(f"Error loading {filename}: {str(e)}", UserWarning)

        def _audio_callback(self, outdata, frames, time, status):
            """SoundDevice callback with underrun handling"""
            if status:
                if status.output_underflow:
                    self.underrun_count += 1
                    warnings.warn(f"Audio underrun detected (total: {self.underrun_count})", UserWarning)
                else:
                    warnings.warn(str(status), UserWarning)
            
            try:
                chunk = self.play_audio_queue.get_nowait()
                if chunk.shape[0] < frames:
                    outdata[:chunk.shape[0]] = chunk
                    outdata[chunk.shape[0]:] = 0
                else:
                    outdata[:] = chunk[:frames]
            except queue.Empty:
                outdata[:] = 0
                if self.stop_playing_audio_event.is_set():
                    raise sd.CallbackStop

        def _load_files(self):
            """Load WAV files from the file queue in a separate thread if not already loading"""
            
            while not self.stop_playing_audio_event.is_set():

                if self.file_queue.empty():
                    break

                try:
                    file = self.file_queue.get()
                    print(f"Loading file: {file}")
                    self._load_wav_chunks(file)

                except queue.Empty:
                    break

                except Exception as e:
                    warnings.warn(f"Error loading file: {file}: {str(e)}", UserWarning)

            return


        def add_file(self, filename: str):
            """Add a WAV file to the load file queue"""
            self.file_queue.put(filename)

            if not self.loading_thread:
                loader = threading.Thread(target=self._load_files)
                loader.start()
                self.loading_thread = True

        def play(self):
            """Start audio playback"""
            if self.play_stream and self.play_stream.active:
                return

            self.stop_playing_audio_event.clear()
            self.play_stream = sd.OutputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                blocksize=self.blocksize,
                callback=self._audio_callback,
                finished_callback=self.stop_playing_audio_event.set
            )
            self.play_stream.start()

        def stop(self):
            """Stop playback immediately (thread-safe)"""
            self.stop_playing_audio_event.set()
            with self.play_queue_lock:
                while not self.play_audio_queue.empty():
                    try:
                        self.play_audio_queue.get_nowait()
                    except queue.Empty:
                        break
            if self.play_stream:
                self.play_stream.abort()
                self.play_stream.close()

        def is_playing(self):
            """Check if playback is active"""
            return self.play_stream and self.play_stream.active

        def get_queue_size(self):
            """Get remaining chunks in queue (thread-safe)"""
            with self.play_queue_lock:
                return self.play_audio_queue.qsize()


# # Example usage

# if __name__ == "__main__":
# 	# Example usage
# 	player = audio_handler,()
    
# 	# Add files to play
# 	player.add_file("dataset/speaker2/odd1out_s2.wav")
# 	player.add_file("dataset/speaker2/odd1out_s2.wav")
# 	player.add_file("dataset/speaker2/odd1out_s2.wav")
    
# 	# Start playback
# 	player.play()

# 	print("Playback started...")
    
# 	try:
# 		while player.is_playing():
# 			print(f"Queue size: {player.get_queue_size()} | Underruns: {player.underrun_count}", end='\r')
# 			time.sleep(0.1)
# 	except KeyboardInterrupt:
# 		player.stop()
# 		print("\nPlayback stopped by user")
# 		exit(0)


recorder = audio_handler.record()
player = audio_handler.play()


def test_parallel_audio():
    print("\nStarting parallel audio test.")
    
    # Create test objects

    try:
        print("Starting recording thread")
        recorder.start_recording()
        time.sleep(1)
        
    except Exception as e:
        print(f"Error in parallel audio test: {e}")
    
    try:
        print("Starting playback thread")
        for _ in range(5):  # Play 5 files
            player.add_file("dataset/speaker2/odd1out_s2.wav")
        print("starting playback...")
        player.play()
        time.sleep(1)
    except Exception as e:
        print(f"Error in parallel audio test: {e}")
    
    print("all threads started")

# Usage example
if __name__ == "__main__":
    try:
        print("Running parallel audio test...")
        test_parallel_audio()
        print("stop program by pressing Ctrl+C")
        while True:
            print(" "*15, end="\r")
            print(f"player queue: {player.get_queue_size()}")

    except KeyboardInterrupt:
        print("\nAll threads stopping")
        recorder.save_recording("test_recording.wav")
        recorder.stop_recording()
        player.stop()
        print("\nTest interrupted by user")
        exit(0)