import subprocess
import queue
import threading
import time
import numpy as np
from faster_whisper import WhisperModel


FFMPEG_PATH = "/tmp/ffmpeg-7.0.2-amd64-static/ffmpeg"
SAMPLE_RATE = 16000
CHUNK_SEC = 5


class AudioProcessor:
    def __init__(self, rtsp_url: str, model_size: str = "base"):
        self.rtsp_url = rtsp_url
        self.text_queue: queue.Queue[str] = queue.Queue()
        self.running = False

        print(f"[INFO] Loading Whisper ({model_size})...")
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def _capture_audio(self):
        cmd = [
            FFMPEG_PATH,
            "-rtsp_transport", "udp",
            "-i", self.rtsp_url,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-f", "wav",
            "-",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return proc

    def _transcribe(self, audio_data: bytes):
        import io
        import wave

        with io.BytesIO() as buf:
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_data)
            buf.seek(0)
            segments, _ = self.model.transcribe(buf, language="ru", vad_filter=True)

            text = " ".join(seg.text for seg in segments).strip()
            if text:
                print(f"[ASR] {text}")
                self.text_queue.put(text)

    def run(self):
        self.running = True
        print("[INFO] Audio capture started (UDP RTSP)")

        while self.running:
            try:
                proc = self._capture_audio()
                buffer = bytearray()

                while self.running:
                    chunk = proc.stdout.read(SAMPLE_RATE * 2 * CHUNK_SEC)
                    if not chunk:
                        break
                    buffer.extend(chunk)

                    if len(buffer) >= SAMPLE_RATE * 2 * CHUNK_SEC:
                        self._transcribe(bytes(buffer))
                        buffer.clear()

                proc.wait()
            except Exception as e:
                print(f"[WARN] Audio error: {e}")
                time.sleep(2)

    def start(self):
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        return thread

    def stop(self):
        self.running = False

    def get_text(self) -> str | None:
        try:
            return self.text_queue.get_nowait()
        except queue.Empty:
            return None
