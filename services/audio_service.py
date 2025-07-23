import wave
import base64
import math
import os
import subprocess
import logging
import json
import time
import struct
from typing import Optional, Tuple
import asyncio

# Try to import audioop for proper μ-law conversion
try:
    import audioop
    HAS_AUDIOOP = True
except ImportError:
    HAS_AUDIOOP = False

logger = logging.getLogger(__name__)

class AudioRecorder:
    def __init__(self, filename: str, sample_rate=8000):
        self.filename = filename
        self.sample_rate = sample_rate
        self.wav_filename = filename.replace('.mp3', '.wav') if filename.endswith('.mp3') else filename
        self.incoming_chunks = []
        self.outgoing_chunks = []
        
    def add_incoming_chunk(self, payload: str):
        """Add incoming RTP payload for conversation recording"""
        try:
            # Decode base64 to get raw μ-law data
            ulaw_data = base64.b64decode(payload)
            
            if HAS_AUDIOOP:
                # Use proper audioop conversion
                linear_data = audioop.ulaw2lin(ulaw_data, 1)  # width=1 for μ-law input
                # Convert to 16-bit for WAV storage
                linear_16bit = audioop.lin2lin(linear_data, 1, 2)  # 8-bit to 16-bit
                self.incoming_chunks.append(linear_16bit)
            else:
                # Use fallback conversion
                linear_samples = []
                for ulaw_byte in ulaw_data:
                    linear_sample = self._fallback_ulaw_to_linear(ulaw_byte)
                    linear_samples.append(struct.pack('<h', linear_sample))
                self.incoming_chunks.append(b''.join(linear_samples))
                
        except Exception as e:
            logger.error(f"Error processing incoming audio: {e}")
        
    def add_outgoing_chunk(self, payload: str):
        """Add outgoing RTP payload for conversation recording"""
        try:
            # Decode base64 to get raw μ-law data
            ulaw_data = base64.b64decode(payload)
            
            if HAS_AUDIOOP:
                # Use proper audioop conversion
                linear_data = audioop.ulaw2lin(ulaw_data, 1)  # width=1 for μ-law input
                # Convert to 16-bit for WAV storage
                linear_16bit = audioop.lin2lin(linear_data, 1, 2)  # 8-bit to 16-bit
                self.outgoing_chunks.append(linear_16bit)
            else:
                # Use fallback conversion
                linear_samples = []
                for ulaw_byte in ulaw_data:
                    linear_sample = self._fallback_ulaw_to_linear(ulaw_byte)
                    linear_samples.append(struct.pack('<h', linear_sample))
                self.outgoing_chunks.append(b''.join(linear_samples))
                
        except Exception as e:
            logger.error(f"Error processing outgoing audio: {e}")
        
    def finalize_recording(self) -> bool:
        """Save audio to file with proper format"""
        if not self.incoming_chunks and not self.outgoing_chunks:
            logger.warning("No audio data to save")
            return False
            
        try:
            # Save incoming audio if available
            if self.incoming_chunks:
                incoming_file = self.wav_filename.replace('.wav', '_incoming.wav')
                self._save_audio_chunks(self.incoming_chunks, incoming_file)
                logger.info(f"✅ Incoming audio saved: {incoming_file} ({len(self.incoming_chunks)} chunks)")
                
            # Save outgoing audio if available
            if self.outgoing_chunks:
                outgoing_file = self.wav_filename.replace('.wav', '_outgoing.wav')
                self._save_audio_chunks(self.outgoing_chunks, outgoing_file)
                logger.info(f"✅ Outgoing audio saved: {outgoing_file} ({len(self.outgoing_chunks)} chunks)")
                
            return True
                
        except Exception as e:
            logger.error(f"❌ Error saving audio recording: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return False
            
    def _save_audio_chunks(self, audio_chunks, filename):
        """Save linear PCM audio data as WAV file"""
        raw_data = b''.join(audio_chunks)
        
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)      # Mono
            wav_file.setsampwidth(2)      # 16-bit linear PCM
            wav_file.setframerate(self.sample_rate)   # 8kHz
            wav_file.writeframes(raw_data)
            
    def _fallback_ulaw_to_linear(self, ulaw):
        """Fallback μ-law decoding when audioop is not available"""
        ulaw = ~ulaw & 0xFF
        sign = ulaw & 0x80
        exponent = (ulaw >> 4) & 0x07
        mantissa = ulaw & 0x0F
        
        sample = ((mantissa << 3) | 0x84) << exponent
        sample -= 0x84
        
        if sign:
            sample = -sample
        
        return max(-32768, min(32767, sample))
    
    def _convert_to_mp3(self):
        """Convert WAV to MP3 using ffmpeg"""
        try:
            subprocess.run(['ffmpeg', '-i', self.wav_filename, '-codec:a', 'mp3', self.filename], 
                         check=True, capture_output=True)
            os.remove(self.wav_filename)  # Clean up WAV
            logger.info(f"✅ Converted to MP3: {self.filename}")
        except subprocess.CalledProcessError:
            logger.warning(f"⚠️ FFmpeg not available, keeping WAV file: {self.wav_filename}")
        except FileNotFoundError:
            logger.warning(f"⚠️ FFmpeg not found, keeping WAV file: {self.wav_filename}")

class AudioPlayer:
    def __init__(self, sample_rate=8000):
        self.sample_rate = sample_rate
        self.is_playing = False
        
    def load_mp3_file(self, filepath: str) -> bool:
        """Load MP3 file and convert to PCMU format for streaming"""
        if not os.path.exists(filepath):
            logger.error(f"Audio file not found: {filepath}")
            return False
            
        try:
            # Convert MP3 to WAV format suitable for PCMU encoding
            temp_wav = filepath.replace('.mp3', '_temp.wav')
            subprocess.run([
                'ffmpeg', '-i', filepath, 
                '-ar', str(self.sample_rate), 
                '-ac', '1', 
                '-f', 'wav', 
                temp_wav
            ], check=True, capture_output=True)
            
            # Load WAV data
            with wave.open(temp_wav, 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                self.audio_data = frames
                
            os.remove(temp_wav)  # Clean up
            logger.info(f"✅ Loaded audio file: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading audio file: {e}")
            return False
    
    async def generate_audio_chunks(self, chunk_size_ms=20):
        """Generate audio chunks from loaded file"""
        if not hasattr(self, 'audio_data'):
            logger.error("No audio data loaded")
            return
            
        # Calculate chunk size in bytes
        chunk_size_bytes = int(self.sample_rate * chunk_size_ms / 1000 * 2)  # 16-bit = 2 bytes per sample
        
        self.is_playing = True
        offset = 0
        
        while offset < len(self.audio_data) and self.is_playing:
            chunk = self.audio_data[offset:offset + chunk_size_bytes]
            if len(chunk) < chunk_size_bytes:
                # Pad with silence if needed
                chunk += b'\x00' * (chunk_size_bytes - len(chunk))
            
            # Convert to PCMU encoding
            pcmu_chunk = self._convert_to_pcmu(chunk)
            yield pcmu_chunk
            
            offset += chunk_size_bytes
            await asyncio.sleep(chunk_size_ms / 1000.0)
            
        self.is_playing = False
        logger.info("Audio playback completed")
    
    def _convert_to_pcmu(self, wav_data: bytes) -> bytes:
        """Convert 16-bit WAV data to PCMU encoding"""
        # Simplified PCMU conversion
        pcmu_data = []
        for i in range(0, len(wav_data), 2):
            if i + 1 < len(wav_data):
                # Convert 16-bit sample to μ-law
                sample = int.from_bytes(wav_data[i:i+2], 'little', signed=True)
                # Simple μ-law approximation
                pcmu_byte = self._linear_to_mulaw(sample)
                pcmu_data.append(bytes([pcmu_byte]))
        
        return b''.join(pcmu_data)
    
    def _linear_to_mulaw(self, sample: int) -> int:
        """Convert 16-bit linear sample to μ-law"""
        # Simplified μ-law conversion
        if sample >= 0:
            return min(127 + int(sample / 256), 255)
        else:
            return max(0, 127 + int(sample / 256))
    
    def stop_playback(self):
        """Stop current playback"""
        self.is_playing = False

class TelnyxAudioParser:
    @staticmethod
    def parse_audio_message(text_data: str) -> Tuple[Optional[bytes], Optional[str]]:
        """Parse Telnyx JSON audio message and extract audio payload"""
        try:
            data = json.loads(text_data)
            if data.get("event") == "media" and "media" in data:
                media = data["media"]
                payload = media.get("payload", "")
                track = media.get("track", "unknown")
                if payload:
                    # Decode base64 audio data
                    audio_data = base64.b64decode(payload)
                    return audio_data, track
        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"Failed to parse audio message: {e}")
        return None, None
    
    @staticmethod
    def create_audio_message(stream_id: str, audio_data: bytes, chunk_number: int) -> str:
        """Create Telnyx audio message from audio data"""
        b64_payload = base64.b64encode(audio_data).decode('utf-8')
        
        message = {
            "event": "media",
            "stream_id": stream_id,
            "media": {
                "track": "outbound", 
                "chunk": str(chunk_number),
                "timestamp": str(chunk_number * 20),  # Sequential timestamp
                "payload": b64_payload
            }
        }
        
        return json.dumps(message)

class ContinuousAudioGenerator:
    """Generate continuous audio for phone calls"""
    
    def __init__(self):
        self.frequencies = [440, 523, 659, 784, 880, 1047]  # A, C, E, G, A, C (nice progression)
        self.current_freq_index = 0
        
    def generate_audio_chunk(self, duration_ms=1000):
        """Generate a continuous audio chunk with rotating frequencies"""
        sample_rate = 8000
        samples = int(sample_rate * duration_ms / 1000.0)
        
        # Get current frequency and advance for next chunk
        frequency = self.frequencies[self.current_freq_index]
        self.current_freq_index = (self.current_freq_index + 1) % len(self.frequencies)
        
        # Generate 16-bit linear sine wave
        linear_samples = []
        for i in range(samples):
            t = i / sample_rate
            # Generate a pleasant sine wave with fade in/out to avoid clicks
            fade = 1.0
            fade_duration = 0.05  # 50ms fade
            if t < fade_duration:
                fade = t / fade_duration
            elif t > (duration_ms/1000.0) - fade_duration:
                fade = ((duration_ms/1000.0) - t) / fade_duration
                
            sine_val = 0.3 * fade * math.sin(2 * math.pi * frequency * t)
            linear_sample = int(sine_val * 32767)
            linear_samples.append(struct.pack('<h', linear_sample))
        
        # Convert to μ-law
        linear_data = b''.join(linear_samples)
        
        if HAS_AUDIOOP:
            # Use proper audioop conversion
            ulaw_data = audioop.lin2ulaw(linear_data, 2)  # 16-bit linear to μ-law
        else:
            # Use fallback conversion
            ulaw_bytes = []
            for i in range(0, len(linear_data), 2):
                if i + 1 < len(linear_data):
                    linear_sample = struct.unpack('<h', linear_data[i:i+2])[0]
                    ulaw_byte = self._fallback_linear_to_ulaw(linear_sample)
                    ulaw_bytes.append(ulaw_byte)
            ulaw_data = bytes(ulaw_bytes)
        
        # Encode as base64
        payload = base64.b64encode(ulaw_data).decode('utf-8')
        
        logger.debug(f"🎵 Generated {duration_ms}ms audio: {frequency}Hz, {samples} samples, {len(ulaw_data)} μ-law bytes")
        
        return payload
        
    def _fallback_linear_to_ulaw(self, sample):
        """Fallback μ-law encoding when audioop is not available"""
        if sample < 0:
            sample = -sample
            sign = 0x80
        else:
            sign = 0x00
        
        if sample > 0x1FFF:
            sample = 0x1FFF
        
        sample += 0x84
        exponent = 7
        for exp_mask in [0x4000, 0x2000, 0x1000, 0x800, 0x400, 0x200, 0x100]:
            if sample >= exp_mask:
                break
            exponent -= 1
        
        mantissa = (sample >> (exponent + 3)) & 0x0F
        ulaw = ~(sign | (exponent << 4) | mantissa)
        return ulaw & 0xFF