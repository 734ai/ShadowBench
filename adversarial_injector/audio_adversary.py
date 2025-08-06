"""
Audio Adversarial Attack Generator for ShadowBench
Implements sophisticated audio-based attacks for multimodal AI systems.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
import io
import base64
import wave
import struct
import json


class AudioAdversary:
    """
    Advanced audio adversarial attack generator for multimodal AI testing.
    
    Implements:
    - Audio noise injection
    - Subliminal message embedding
    - Frequency domain attacks
    - Echo and reverb manipulation
    - Adversarial perturbations
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Standard audio parameters
        self.sample_rate = 44100
        self.channels = 1
        self.sample_width = 2  # 16-bit
        
        # Attack parameters
        self.noise_levels = {
            'low': 0.001,
            'medium': 0.01,
            'high': 0.05
        }
        
        # Frequency bands for attacks
        self.frequency_bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250),
            'low_mid': (250, 500),
            'mid': (500, 2000),
            'high_mid': (2000, 4000),
            'presence': (4000, 6000),
            'brilliance': (6000, 20000)
        }
    
    def generate_adversarial_audio(self, audio_data: bytes, 
                                 attack_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate adversarial version of input audio.
        
        Args:
            audio_data: Raw audio bytes
            attack_config: Configuration for attacks to apply
            
        Returns:
            Dictionary containing adversarial audio data and metadata
        """
        try:
            # Parse audio data
            audio_array = self._parse_audio_bytes(audio_data)
            if audio_array is None:
                return {'success': False, 'error': 'Failed to parse audio data'}
            
            adversarial_audio = audio_array.copy()
            attacks_applied = []
            
            # Apply configured attacks
            for attack_name, attack_params in attack_config.items():
                if attack_params.get('enabled', False):
                    if hasattr(self, f'_apply_{attack_name}'):
                        attack_func = getattr(self, f'_apply_{attack_name}')
                        adversarial_audio = attack_func(adversarial_audio, attack_params)
                        attacks_applied.append(attack_name)
                        self.logger.debug(f"Applied {attack_name} attack")
            
            # Convert back to bytes
            adversarial_bytes = self._audio_array_to_bytes(adversarial_audio)
            adversarial_b64 = base64.b64encode(adversarial_bytes).decode()
            
            # Calculate perturbation metrics
            perturbation_metrics = self._calculate_audio_perturbation_metrics(
                audio_array, adversarial_audio
            )
            
            return {
                'adversarial_audio_b64': adversarial_b64,
                'attacks_applied': attacks_applied,
                'perturbation_metrics': perturbation_metrics,
                'duration_seconds': len(adversarial_audio) / self.sample_rate,
                'sample_rate': self.sample_rate,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate adversarial audio: {e}")
            return {
                'success': False,
                'error': str(e),
                'attacks_applied': [],
                'perturbation_metrics': {}
            }
    
    def _parse_audio_bytes(self, audio_data: bytes) -> Optional[np.ndarray]:
        """Parse audio bytes into numpy array."""
        try:
            # Try to parse as WAV first
            if audio_data.startswith(b'RIFF'):
                return self._parse_wav_bytes(audio_data)
            else:
                # Assume raw PCM data
                return np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as e:
            self.logger.error(f"Failed to parse audio: {e}")
            return None
    
    def _parse_wav_bytes(self, wav_data: bytes) -> np.ndarray:
        """Parse WAV bytes into numpy array."""
        with io.BytesIO(wav_data) as wav_io:
            with wave.open(wav_io, 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                if wav_file.getsampwidth() == 2:
                    audio_array = np.frombuffer(frames, dtype=np.int16)
                elif wav_file.getsampwidth() == 4:
                    audio_array = np.frombuffer(frames, dtype=np.int32)
                else:
                    audio_array = np.frombuffer(frames, dtype=np.uint8)
                
                # Convert to float32 in range [-1, 1]
                max_val = 2 ** (wav_file.getsampwidth() * 8 - 1)
                return audio_array.astype(np.float32) / max_val
    
    def _audio_array_to_bytes(self, audio_array: np.ndarray) -> bytes:
        """Convert audio array to WAV bytes."""
        # Convert float32 back to int16
        int16_audio = (audio_array * 32767).astype(np.int16)
        
        # Create WAV file in memory
        with io.BytesIO() as wav_io:
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(self.sample_width)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(int16_audio.tobytes())
            
            return wav_io.getvalue()
    
    def _apply_noise_injection(self, audio: np.ndarray, params: Dict) -> np.ndarray:
        """Apply noise injection attack."""
        noise_level = params.get('level', 'medium')
        noise_type = params.get('type', 'gaussian')
        target_frequency = params.get('target_frequency', None)
        
        if noise_type == 'gaussian':
            noise = np.random.normal(0, self.noise_levels[noise_level], audio.shape)
        elif noise_type == 'uniform':
            noise = np.random.uniform(-self.noise_levels[noise_level], 
                                    self.noise_levels[noise_level], audio.shape)
        elif noise_type == 'pink':
            noise = self._generate_pink_noise(len(audio), self.noise_levels[noise_level])
        elif noise_type == 'brown':
            noise = self._generate_brown_noise(len(audio), self.noise_levels[noise_level])
        else:
            noise = np.random.normal(0, self.noise_levels[noise_level], audio.shape)
        
        # Apply frequency-specific noise if specified
        if target_frequency:
            noise = self._apply_frequency_filter(noise, target_frequency)
        
        return np.clip(audio + noise, -1.0, 1.0)
    
    def _apply_subliminal_embedding(self, audio: np.ndarray, params: Dict) -> np.ndarray:
        """Apply subliminal message embedding."""
        message = params.get('message', 'HIDDEN MESSAGE')
        frequency = params.get('frequency', 440)  # Hz
        amplitude = params.get('amplitude', 0.01)
        modulation = params.get('modulation', 'amplitude')
        
        # Generate carrier signal
        t = np.linspace(0, len(audio) / self.sample_rate, len(audio), False)
        carrier = np.sin(2 * np.pi * frequency * t)
        
        if modulation == 'amplitude':
            # Amplitude modulation
            message_signal = self._text_to_signal(message, len(audio))
            modulated = carrier * (1 + amplitude * message_signal)
        elif modulation == 'frequency':
            # Frequency modulation
            message_signal = self._text_to_signal(message, len(audio))
            freq_dev = 50  # Hz
            phase = 2 * np.pi * np.cumsum(frequency + freq_dev * message_signal) / self.sample_rate
            modulated = np.sin(phase)
        else:
            modulated = carrier
        
        # Embed in audio
        embedded_audio = audio + amplitude * modulated
        return np.clip(embedded_audio, -1.0, 1.0)
    
    def _apply_frequency_attack(self, audio: np.ndarray, params: Dict) -> np.ndarray:
        """Apply frequency domain attack."""
        attack_type = params.get('type', 'boost')
        target_band = params.get('band', 'mid')
        intensity = params.get('intensity', 0.1)
        
        # Get frequency band
        if target_band in self.frequency_bands:
            low_freq, high_freq = self.frequency_bands[target_band]
        else:
            low_freq, high_freq = 500, 2000
        
        # Apply FFT
        fft_audio = np.fft.rfft(audio)
        frequencies = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
        
        # Find target frequency indices
        band_mask = (frequencies >= low_freq) & (frequencies <= high_freq)
        
        if attack_type == 'boost':
            fft_audio[band_mask] *= (1 + intensity)
        elif attack_type == 'suppress':
            fft_audio[band_mask] *= (1 - intensity)
        elif attack_type == 'phase_shift':
            phase_shift = intensity * np.pi
            fft_audio[band_mask] *= np.exp(1j * phase_shift)
        elif attack_type == 'invert':
            fft_audio[band_mask] = -fft_audio[band_mask]
        
        # Convert back to time domain
        modified_audio = np.fft.irfft(fft_audio, len(audio))
        return np.clip(modified_audio, -1.0, 1.0)
    
    def _apply_echo_manipulation(self, audio: np.ndarray, params: Dict) -> np.ndarray:
        """Apply echo and reverb manipulation."""
        delay_ms = params.get('delay_ms', 100)
        decay_factor = params.get('decay_factor', 0.3)
        num_echoes = params.get('num_echoes', 3)
        reverb_type = params.get('reverb_type', 'simple')
        
        delay_samples = int(delay_ms * self.sample_rate / 1000)
        result = audio.copy()
        
        if reverb_type == 'simple':
            # Simple echo effect
            for i in range(1, num_echoes + 1):
                echo_delay = delay_samples * i
                if echo_delay < len(audio):
                    echo_amplitude = decay_factor ** i
                    echo_start = min(echo_delay, len(result))
                    echo_end = min(len(audio), len(result))
                    
                    if echo_start < echo_end:
                        result[echo_start:echo_end] += echo_amplitude * audio[:echo_end - echo_start]
        
        elif reverb_type == 'room':
            # Room reverb simulation
            reverb_time = params.get('reverb_time', 0.5)  # seconds
            reverb_samples = int(reverb_time * self.sample_rate)
            
            # Generate impulse response for room reverb
            impulse_response = self._generate_room_impulse_response(reverb_samples, decay_factor)
            
            # Convolve with audio (simplified)
            result = np.convolve(audio, impulse_response, mode='same')
        
        return np.clip(result, -1.0, 1.0)
    
    def _apply_adversarial_perturbation(self, audio: np.ndarray, params: Dict) -> np.ndarray:
        """Apply targeted adversarial perturbations."""
        perturbation_type = params.get('type', 'targeted')
        epsilon = params.get('epsilon', 0.01)
        target_class = params.get('target_class', None)
        
        # Generate adversarial perturbation
        if perturbation_type == 'targeted':
            # Targeted attack - push towards specific class
            perturbation = self._generate_targeted_perturbation(audio, target_class, epsilon)
        elif perturbation_type == 'untargeted':
            # Untargeted attack - maximize confusion
            perturbation = self._generate_untargeted_perturbation(audio, epsilon)
        elif perturbation_type == 'imperceptible':
            # Highly imperceptible perturbation
            perturbation = self._generate_imperceptible_perturbation(audio, epsilon)
        else:
            perturbation = np.random.normal(0, epsilon, audio.shape)
        
        return np.clip(audio + perturbation, -1.0, 1.0)
    
    def _generate_pink_noise(self, length: int, amplitude: float) -> np.ndarray:
        """Generate pink noise (1/f noise)."""
        # Generate white noise
        white_noise = np.random.normal(0, 1, length)
        
        # Apply pink noise filter (approximation)
        # Pink noise has power spectral density proportional to 1/f
        fft_noise = np.fft.rfft(white_noise)
        frequencies = np.fft.rfftfreq(length, 1/self.sample_rate)
        frequencies[0] = 1  # Avoid division by zero
        
        # Apply 1/f^0.5 scaling (pink noise)
        pink_filter = 1 / np.sqrt(frequencies)
        pink_fft = fft_noise * pink_filter
        
        pink_noise = np.fft.irfft(pink_fft, length)
        
        # Normalize and scale
        pink_noise = pink_noise / np.std(pink_noise) * amplitude
        return pink_noise
    
    def _generate_brown_noise(self, length: int, amplitude: float) -> np.ndarray:
        """Generate brown noise (1/f^2 noise)."""
        # Generate white noise
        white_noise = np.random.normal(0, 1, length)
        
        # Apply brown noise filter
        fft_noise = np.fft.rfft(white_noise)
        frequencies = np.fft.rfftfreq(length, 1/self.sample_rate)
        frequencies[0] = 1  # Avoid division by zero
        
        # Apply 1/f scaling (brown noise)
        brown_filter = 1 / frequencies
        brown_fft = fft_noise * brown_filter
        
        brown_noise = np.fft.irfft(brown_fft, length)
        
        # Normalize and scale
        brown_noise = brown_noise / np.std(brown_noise) * amplitude
        return brown_noise
    
    def _apply_frequency_filter(self, signal: np.ndarray, target_freq: float) -> np.ndarray:
        """Apply frequency-specific filtering."""
        fft_signal = np.fft.rfft(signal)
        frequencies = np.fft.rfftfreq(len(signal), 1/self.sample_rate)
        
        # Create bandpass filter around target frequency
        bandwidth = 100  # Hz
        filter_mask = np.abs(frequencies - target_freq) <= bandwidth
        
        # Apply filter
        fft_signal[~filter_mask] = 0
        
        return np.fft.irfft(fft_signal, len(signal))
    
    def _text_to_signal(self, text: str, length: int) -> np.ndarray:
        """Convert text to signal using ASCII values."""
        # Convert text to repeating signal pattern
        ascii_values = [ord(c) for c in text]
        normalized_values = [(v - 64) / 32 for v in ascii_values]  # Normalize ASCII
        
        # Repeat pattern to match audio length
        pattern_length = len(normalized_values)
        repetitions = length // pattern_length + 1
        signal = np.tile(normalized_values, repetitions)[:length]
        
        return np.array(signal)
    
    def _generate_room_impulse_response(self, length: int, decay: float) -> np.ndarray:
        """Generate room impulse response for reverb."""
        # Simple exponential decay with random reflections
        impulse = np.zeros(length)
        
        # Direct path
        impulse[0] = 1.0
        
        # Early reflections
        num_reflections = min(10, length // 100)
        for i in range(num_reflections):
            delay = np.random.randint(50, length // 4)
            amplitude = decay ** (i + 1) * np.random.uniform(0.5, 1.0)
            if delay < length:
                impulse[delay] += amplitude
        
        # Late reverb (exponential decay)
        for i in range(length):
            if i > 0:
                impulse[i] += impulse[i-1] * decay * 0.99
        
        return impulse / np.max(np.abs(impulse))
    
    def _generate_targeted_perturbation(self, audio: np.ndarray, target_class: str, 
                                      epsilon: float) -> np.ndarray:
        """Generate targeted adversarial perturbation."""
        # Simplified targeted perturbation
        # In practice, this would use gradient-based methods
        
        if target_class == 'speech':
            # Add speech-like formants
            perturbation = self._add_formant_like_patterns(audio, epsilon)
        elif target_class == 'music':
            # Add harmonic patterns
            perturbation = self._add_harmonic_patterns(audio, epsilon)
        elif target_class == 'silence':
            # Reduce overall amplitude
            perturbation = -audio * epsilon
        else:
            # Random perturbation
            perturbation = np.random.normal(0, epsilon, audio.shape)
        
        return perturbation
    
    def _generate_untargeted_perturbation(self, audio: np.ndarray, epsilon: float) -> np.ndarray:
        """Generate untargeted adversarial perturbation."""
        # Generate high-frequency noise to maximize confusion
        perturbation = np.random.normal(0, epsilon, audio.shape)
        
        # Emphasize high frequencies
        fft_pert = np.fft.rfft(perturbation)
        frequencies = np.fft.rfftfreq(len(perturbation), 1/self.sample_rate)
        
        # Boost high frequencies
        high_freq_mask = frequencies > 5000
        fft_pert[high_freq_mask] *= 2
        
        perturbation = np.fft.irfft(fft_pert, len(perturbation))
        
        return perturbation
    
    def _generate_imperceptible_perturbation(self, audio: np.ndarray, epsilon: float) -> np.ndarray:
        """Generate highly imperceptible adversarial perturbation."""
        # Use psychoacoustic masking principles
        perturbation = np.random.normal(0, epsilon * 0.1, audio.shape)
        
        # Apply masking based on audio content
        fft_audio = np.fft.rfft(audio)
        fft_pert = np.fft.rfft(perturbation)
        
        # Reduce perturbation in quiet frequency bands
        magnitude_threshold = np.percentile(np.abs(fft_audio), 25)
        quiet_mask = np.abs(fft_audio) < magnitude_threshold
        fft_pert[quiet_mask] *= 0.1
        
        perturbation = np.fft.irfft(fft_pert, len(perturbation))
        
        return perturbation
    
    def _add_formant_like_patterns(self, audio: np.ndarray, epsilon: float) -> np.ndarray:
        """Add speech formant-like patterns."""
        # Typical speech formants: F1=730Hz, F2=1090Hz, F3=2440Hz
        formants = [730, 1090, 2440]
        perturbation = np.zeros_like(audio)
        
        t = np.linspace(0, len(audio) / self.sample_rate, len(audio), False)
        
        for formant_freq in formants:
            # Add formant with slight amplitude modulation
            modulation = 1 + 0.1 * np.sin(2 * np.pi * 5 * t)  # 5Hz modulation
            formant_signal = np.sin(2 * np.pi * formant_freq * t) * modulation
            perturbation += epsilon * 0.3 * formant_signal
        
        return perturbation
    
    def _add_harmonic_patterns(self, audio: np.ndarray, epsilon: float) -> np.ndarray:
        """Add harmonic patterns typical of music."""
        # Add harmonic series based on fundamental frequency
        fundamental = 220  # A3
        harmonics = [1, 2, 3, 4, 5, 6]
        
        perturbation = np.zeros_like(audio)
        t = np.linspace(0, len(audio) / self.sample_rate, len(audio), False)
        
        for harmonic in harmonics:
            freq = fundamental * harmonic
            amplitude = 1.0 / harmonic  # Decreasing amplitude for higher harmonics
            harmonic_signal = np.sin(2 * np.pi * freq * t) * amplitude
            perturbation += epsilon * 0.2 * harmonic_signal
        
        return perturbation
    
    def _calculate_audio_perturbation_metrics(self, original: np.ndarray, 
                                            adversarial: np.ndarray) -> Dict[str, float]:
        """Calculate audio perturbation quality metrics."""
        # Signal-to-Noise Ratio
        signal_power = np.mean(original ** 2)
        noise_power = np.mean((original - adversarial) ** 2)
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
        # Mean Squared Error
        mse = np.mean((original - adversarial) ** 2)
        
        # L2 norm of perturbation
        l2_norm = np.linalg.norm(original - adversarial)
        
        # L_infinity norm
        l_inf_norm = np.max(np.abs(original - adversarial))
        
        # Spectral distance
        spectral_distance = self._calculate_spectral_distance(original, adversarial)
        
        # Perceptual similarity (simplified)
        perceptual_sim = self._calculate_perceptual_similarity(original, adversarial)
        
        return {
            'snr_db': float(snr),
            'mse': float(mse),
            'l2_norm': float(l2_norm),
            'l_inf_norm': float(l_inf_norm),
            'spectral_distance': float(spectral_distance),
            'perceptual_similarity': float(perceptual_sim),
            'perturbation_strength': float(l2_norm / len(original) ** 0.5)
        }
    
    def _calculate_spectral_distance(self, audio1: np.ndarray, audio2: np.ndarray) -> float:
        """Calculate spectral distance between two audio signals."""
        fft1 = np.abs(np.fft.rfft(audio1))
        fft2 = np.abs(np.fft.rfft(audio2))
        
        # Normalize
        fft1 = fft1 / (np.sum(fft1) + 1e-10)
        fft2 = fft2 / (np.sum(fft2) + 1e-10)
        
        # Calculate KL divergence (simplified)
        kl_div = np.sum(fft1 * np.log((fft1 + 1e-10) / (fft2 + 1e-10)))
        
        return kl_div
    
    def _calculate_perceptual_similarity(self, audio1: np.ndarray, audio2: np.ndarray) -> float:
        """Calculate perceptual similarity (simplified)."""
        # Use correlation as proxy for perceptual similarity
        correlation = np.corrcoef(audio1, audio2)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def generate_attack_report(self, attack_results: Dict[str, Any]) -> str:
        """Generate comprehensive audio attack report."""
        if not attack_results.get('success'):
            return f"Audio attack failed: {attack_results.get('error', 'Unknown error')}"
        
        report = f"""
Audio Adversarial Attack Report
===============================

Attacks Applied: {', '.join(attack_results['attacks_applied'])}
Duration: {attack_results['duration_seconds']:.2f} seconds
Sample Rate: {attack_results['sample_rate']} Hz

Perturbation Metrics:
- SNR: {attack_results['perturbation_metrics'].get('snr_db', 0):.2f} dB
- MSE: {attack_results['perturbation_metrics'].get('mse', 0):.6f}
- L2 Norm: {attack_results['perturbation_metrics'].get('l2_norm', 0):.4f}
- L∞ Norm: {attack_results['perturbation_metrics'].get('l_inf_norm', 0):.4f}
- Spectral Distance: {attack_results['perturbation_metrics'].get('spectral_distance', 0):.4f}
- Perceptual Similarity: {attack_results['perturbation_metrics'].get('perceptual_similarity', 0):.4f}

Attack Quality Assessment:
- High imperceptibility: {'Yes' if attack_results['perturbation_metrics'].get('snr_db', 0) > 20 else 'No'}
- Low spectral distortion: {'Yes' if attack_results['perturbation_metrics'].get('spectral_distance', 100) < 1.0 else 'No'}
- Successful generation: {'Yes' if attack_results['success'] else 'No'}
        """
        
        return report.strip()
