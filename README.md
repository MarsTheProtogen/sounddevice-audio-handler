# NOTICE
## THIS LIBRARY IS NOT RELEASED YET 


# 🎵 Audio Handler Documentation 🎶

### Overview
📌 This script provides functionality for **recording**, **playing**, and **processing** audio using the `sounddevice` and `librosa` libraries. It includes classes for handling **real-time audio input/output**, **audio conversion**, and **temporary file management**.

## 🚀 Why Use This Library?
✅ **Real-time audio processing** with efficient recording & playback. <br>
✅ **Automatic sample rate detection** and volume control. <br>
✅ **Chunk-based audio storage** for optimized performance. <br>
✅ Ideal for **audio streaming applications, voice recorders, and audio processing tools**. <br>

# dependencies 
requires `sounddevice
numpy
librosa
scipy` <br>
to install requiremants 
`pip install -r PATH/TO/requrements.txt`

---

# 📖 Documentation

## 🎛 Class: `AudioConverter`
### 🔹 Purpose
Handles loading, resampling, and converting WAV audio files to a desired sample rate, number of channels, and data type.

### 📥 Importing
```python
from audio_handler import AudioConverter
```

### 🛠 Methods
- `__init__(target_samplerate, target_channels, target_dtype=np.int16, chunk_size=1024)`: Initializes the converter with desired properties.
- `load_and_convert_wav(filename)`: Loads and converts a WAV file in chunks.
- `resample_audio(data, input_samplerate, target_samplerate)`: Resamples audio using `librosa`.
- `convert_channels(data, input_channels, target_channels)`: Converts mono to stereo and vice versa.
- `scale_to_target_dtype(data)`: Converts audio to the target data type (int16, float32, etc.).

### 📌 Example Usage
```python
converter = AudioConverter(target_samplerate=44100, target_channels=2, target_dtype=np.float32)
for chunk in converter.load_and_convert_wav("example.wav"):
    print(chunk.dtype)
```

---

## 🎙 Class: `audio_handler.record`

### 📥 Importing
```python
from audio_handler import audio_handler
```

### 🔹 Purpose
Manages **real-time audio recording**, temporary storage, and saving to a file.

### 🛠 Methods
- `detect_sample_rate() -> int`: Detects the sample rate of the default input device.
- `set_volume(volume: int) -> None`: Sets the volume multiplier.
- `start_recording() -> None`: Starts recording audio.
- `stop_recording() -> None`: Stops recording and cleans up temp files.
- `save_recording(filename: str) -> None`: Saves recorded audio to a WAV file.

### 📌 Example Usage
```python
recorder = audio_handler.record()
recorder.start_recording()
time.sleep(5)  # Record for 5 seconds
recorder.save_recording("output.wav")
recorder.stop_recording()
```

---

## 🔊 Class: `audio_handler.play`
### 🔹 Purpose
Handles **real-time audio playback** from file or memory.

### 🛠 Methods
- `add_file(filename: str) -> None`: Adds a WAV file to the play queue.
- `play() -> None`: Starts playing queued audio.
- `stop() -> None`: Stops playback immediately.
- `is_playing() -> bool`: Checks if playback is active.
- `get_queue_size() -> int`: Returns the number of audio chunks in the queue.

### 📌 Example Usage
```python
player = audio_handler.play()
player.add_file("file.wav")
player.play()
time.sleep(10)  # Allow playback for 10 seconds
player.stop()  # Instantly stop playing audio
```

---

## 🎵 Testing Parallel Audio

The script includes a function to test **simultaneous recording and playback**.

```python
import time
from audio_handler import audio_handler

recorder = audio_handler.record()
player = audio_handler.play()

def test_parallel_audio():
    print("\nStarting parallel audio test.")
    recorder.start_recording()
    time.sleep(1)
    for _ in range(5):  # Play 5 files
        player.add_file("example.wav")
    player.play()
    time.sleep(1)

test_parallel_audio()
```

🛑 **To stop execution, press `Ctrl+C`.**

---

## ⚡ Notes
- **🔧 Dependencies:** Ensure `sounddevice`, `numpy`, `librosa`, and `scipy` are installed.
- **⚠️ Error Handling:** Uses `warnings.warn()` for non-fatal errors.
- **📂 Temporary Storage:** Recorded audio is stored in a temporary folder and deleted upon stopping recording.
- **📈 Performance Considerations:** Saving recorded audio in chunks reduces memory usage.

📝 **This documentation provides a comprehensive guide for using the `audio_handler.py` script effectively.** 🎼

<br><br><br><br><br><br><br><br>
