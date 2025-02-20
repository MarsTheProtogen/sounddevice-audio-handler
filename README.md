# sounddevice-audio-handler
A library that simplifies the use of sounddevice's parelell I/O

# Documentation

## Overview
This script provides functionality for recording, playing, and processing audio using the `sounddevice` and `librosa` libraries. It includes classes for handling real-time audio input/output, audio conversion, and temporary file management.

---

## Class: `AudioConverter`
### Purpose
Handles loading, resampling, and converting WAV audio files to a desired sample rate, number of channels, and data type.

### importing
import with `from audio_handler import AudioConverter`

### Methods
- `__init__(target_samplerate, target_channels, target_dtype=np.int16, chunk_size=1024)`: Initializes the converter with desired properties.
- `load_and_convert_wav(filename)`: Loads and converts a WAV file in chunks.
- `resample_audio(data, input_samplerate, target_samplerate)`: Resamples audio using `librosa`.
- `convert_channels(data, input_channels, target_channels)`: Converts mono to stereo and vice versa.
- `scale_to_target_dtype(data)`: Converts audio to the target data type (int16, float32, etc.).

### Example Usage
```python
converter = AudioConverter(target_samplerate=44100, target_channels=2, target_dtype=np.float32)
for chunk in converter.load_and_convert_wav("example.wav"):
    print(chunk.dtype)
```

---

## Class: `audio_handler.record`

### importing
import with `from audio_handler import audio_handler`

### Purpose
Manages real-time audio recording, temporary storage, and saving to a file.

### Methods
- `detect_sample_rate() -> int`: Detects the sample rate of the default input device.
- `set_volume(volume: int) -> None`: Sets the volume multiplier.
- `start_recording() -> None`: Starts recording audio.
- `stop_recording() -> None`: Stops recording and cleans up temp files.
- `save_recording(filename: str) -> None`: Saves recorded audio to a WAV file.
- `_callback(indata, frames, time, status)`: Internal method for handling incoming audio data.
- `_save_audio_chunk() -> None`: Saves recorded chunks to temp files.
- `_temp_folder_cleanup() -> None`: Cleans up temporary storage.

### Example Usage
```python
from audio_handler import audio_handler

recorder = audio_handler.record()
recorder.start_recording()
time.sleep(5)  # Record for 5 seconds

# data should be saved, then closed out.

# saving will start a new thread to save the file,
#allowing .stop_recording() to stop the recording stream
recorder.save_recording("output.wav")
recorder.stop_recording()
```

---

## Class: `audio_handler.play`
### Purpose
Handles real-time audio playback from file or memory.

### Methods
- `get_output_format(device_id: int = None) -> dict`: Retrieves output format details.
- `add_file(filename: str) -> None`: Adds a WAV file to the play queue.
- `play() -> None`: Starts playing queued audio.
- `stop() -> None`: Stops playback immediately.
- `is_playing() -> bool`: Checks if playback is active.
- `get_queue_size() -> int`: Returns the number of audio chunks in the queue.
- `_audio_callback(outdata, frames, time, status)`: Internal method for handling playback output.
- `_load_wav_chunks(filename: str) -> None`: Loads WAV files in chunks.
- `_load_files() -> None`: Loads files into the playback queue.

### Example Usage
```python
from audio_handler import audio_handler

player.add_file("file.wav")
player.play()
player.add_file("file.wav") #queue up another file
time.sleep(10)  # Allow playback for 10 seconds
player.stop() # instantly stop playing audio (will also empty play queue)
```

---

## Testing Parallel Audio
The script includes a function to test simultaneous recording and playback.

```python
import time
from audio_handler import audio_handler

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

if __name__ == "__main__":
    try:
        print("Running parallel audio test...")
        test_parallel_audio()
        print("stop program by pressing Ctrl+C")
        while True:
            print(" "*15, end="\r")
            print(f"player queue: {player.get_queue_size()}")
            time.wait(.1)

    except KeyboardInterrupt:
        print("\nAll threads stopping")
        recorder.save_recording("test_recording.wav")
        recorder.stop_recording()
        player.stop()
        print("\nTest interrupted by user")
        exit(0)
```

To stop execution, press `Ctrl+C`.

---

## Notes
- **Dependencies:** Ensure `sounddevice`, `numpy`, `librosa`, and `scipy` are installed.
- **Error Handling:** Uses `warnings.warn()` for non-fatal errors.
- **Temporary Storage:** Recorded audio is stored in a temporary folder and deleted upon stopping recording.
- **Performance Considerations:** Saving recorded audio in chunks reduces memory usage.

This documentation provides a  guide for using the `audio_handler.py` script effectively.

