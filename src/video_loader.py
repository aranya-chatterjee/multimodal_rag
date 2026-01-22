
import os
import tempfile
import warnings
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

class TranscriptSource(Enum):
    """Enum for transcript sources"""
    WHISPER_AI = "whisper_ai"
    ERROR = "error"

@dataclass
class VideoFileInfo:
# getting the video file information
    filename: str
    filepath: str
    filesize: int
    duration: Optional[float] = None
    transcript_source: Optional[TranscriptSource] = None
    transcript_text: Optional[str] = None
    has_transcript: bool = False

class VideoProcessor:

    def __init__(self, whisper_model: str = "base", language: str = "english"):

        self.whisper_model = whisper_model
        self.language = language
        self.features = {
            'whisper': self._check_whisper_availability(),
            'moviepy': self._check_moviepy_availability()
        }
        self._print_available_features()

    def _check_whisper_availability(self) -> bool:

        try:
            import whisper
            return True
        except ImportError:
            print(" Whisper is not installed ")
            return False

    def _check_moviepy_availability(self) -> bool:

        try:
            import moviepy
            return True
        except ImportError:
            print("Moviepy is not installed ")
            return False

    def _print_available_features(self):
        """Print available features"""
        print("[INFO] Video Processor Features:")
        for feature, available in self.features.items():
            status = "[OK]" if available else "[FAIL]"
            print(f"  {status} {feature}")

    def get_video_info(self, filepath: str) -> Optional[VideoFileInfo]:

        try:
            if not os.path.exists(filepath):
                print(f"[FAIL] File not found: {filepath}")
                return None

            filename = os.path.basename(filepath)
            filesize = os.path.getsize(filepath)

            # Get duration if moviepy is available
            duration = None
            if self.features['moviepy']:
                try:
                    from moviepy.editor import VideoFileClip
                    with VideoFileClip(filepath) as video:
                        duration = video.duration
                except:
                    pass

            return VideoFileInfo(
                filename=filename,
                filepath=filepath,
                filesize=filesize,
                duration=duration
            )

        except Exception as e:
            print(f" Error getting video info: {e}")
            return None

    def extract_audio_from_video(self, video_path: str, audio_path: str) -> bool:

        try:
            if self.features['moviepy']:
                from moviepy import VideoFileClip

                print("[INFO] Extracting audio from video...")
                video = VideoFileClip(video_path)
                print(f"[INFO] Video loaded, duration: {video.duration}s")
                if video.audio:
                    print(f"[INFO] Audio track found, writing to {audio_path}...")
                    video.audio.write_audiofile(audio_path)
                    print("[SUCCESS] Audio extracted successfully")
                    video.close()
                    return True
                else:
                    print("[ERROR] No audio track found in video")
                    video.close()
                    return False
            else:
                print("[ERROR] MoviePy not available for audio extraction")
                return False

        except Exception as e:
            print(f"[ERROR] Error extracting audio: {e}")
            import traceback
            traceback.print_exc()
            return False

    def transcribe_with_whisper(self, audio_path: str) -> Tuple[Optional[str], TranscriptSource]:

        if not self.features['whisper']:
            return None, TranscriptSource.ERROR

        try:
            import whisper
            import librosa
            import numpy as np

            print(f"[AI] Transcribing with Whisper ({self.whisper_model})...")

            # Load Whisper model
            model = whisper.load_model(self.whisper_model)

            # Load audio using librosa instead of relying on ffmpeg
            print("[INFO] Loading audio file...")
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Set language parameter
            language_param = 'en' if self.language.lower() == 'english' else None

            # Transcribe using raw audio
            print("[INFO] Running transcription...")
            result = model.transcribe(
                audio=audio,
                language=language_param,
                task='transcribe'
            )

            print("[SUCCESS] Transcription successful!")
            return result['text'], TranscriptSource.WHISPER_AI

        except Exception as e:
            import traceback
            print(f"[ERROR] Whisper transcription failed: {e}")
            print(f"[ERROR] Error details: {traceback.format_exc()}")
            return None, TranscriptSource.ERROR

    def process_video_file(self, filepath: str) -> Tuple[Optional[List[Dict]], VideoFileInfo]:

        from langchain_core.documents import Document

        print(f"\n[VIDEO] Processing video file: {os.path.basename(filepath)}")

        # Get video info
        video_info = self.get_video_info(filepath)
        if not video_info:
            print(" Could not get video information")
            return None, None

        # Check if file is MP4
        if not filepath.lower().endswith('.mp4'):
            print(f" Warning: {filepath} is not an MP4 file. Trying anyway...")

        # Create temp directory for audio extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = os.path.join(temp_dir, "extracted_audio.wav")

            # Extract audio
            if not self.extract_audio_from_video(filepath, audio_path):
                print("Failed to extract audio")
                video_info.transcript_source = TranscriptSource.ERROR
                video_info.has_transcript = False

                # Create minimal document
                documents = [{
                    'page_content': f"Video file: {video_info.filename}\nFailed to extract audio for transcription.",
                    'metadata': {
                        'source_type': 'video',
                        'file_name': video_info.filename,
                        'file_path': filepath,
                        'file_size': video_info.filesize,
                        'duration': video_info.duration,
                        'transcript_source': TranscriptSource.ERROR.value,
                        'has_transcript': False,
                        'language': self.language
                    }
                }]
                return documents, video_info

            # Transcribe audio
            transcript_text, transcript_source = self.transcribe_with_whisper(audio_path)

            # Clean up temp audio file
            try:
                os.remove(audio_path)
            except:
                pass

            # Update video info
            video_info.transcript_source = transcript_source
            video_info.transcript_text = transcript_text
            video_info.has_transcript = transcript_source == TranscriptSource.WHISPER_AI

            # Create documents
            if transcript_text:
                documents = [{
                    'page_content': transcript_text,
                    'metadata': {
                        'source_type': 'video',
                        'file_name': video_info.filename,
                        'file_path': filepath,
                        'file_size': video_info.filesize,
                        'duration': video_info.duration,
                        'transcript_source': transcript_source.value,
                        'whisper_model': self.whisper_model,
                        'has_transcript': True,
                        'language': self.language,
                        'word_count': len(transcript_text.split())
                    }
                }]

                print(f"[OK] Processed successfully!")
                print(f"   Transcript length: {len(transcript_text)} characters")
                print(f"   Word count: {len(transcript_text.split())}")

            else:
                # Create document with error info
                documents = [{
                    'page_content': f"Video file: {video_info.filename}\nFailed to transcribe audio content.",
                    'metadata': {
                        'source_type': 'video',
                        'file_name': video_info.filename,
                        'file_path': filepath,
                        'file_size': video_info.filesize,
                        'duration': video_info.duration,
                        'transcript_source': TranscriptSource.ERROR.value,
                        'has_transcript': False,
                        'language': self.language
                    }
                }]
                print(" Transcription failed")

        return documents, video_info

    def batch_process_videos(self, filepaths: List[str]) -> Dict[str, Tuple[List[Dict], VideoFileInfo]]:

        results = {}

        for i, filepath in enumerate(filepaths):
            print(f"\n[STATS] Processing video {i+1}/{len(filepaths)}")
            documents, video_info = self.process_video_file(filepath)
            results[filepath] = (documents, video_info)

        return results

# Convenience function
def load_video_files(filepaths: List[str], whisper_model: str = "base",
                     language: str = "english") -> Dict[str, Any]:

    processor = VideoProcessor(whisper_model=whisper_model, language=language)

    if isinstance(filepaths, str):
        filepaths = [filepaths]

    results = processor.batch_process_videos(filepaths)

    # Format results
    formatted_results = {
        'success_count': 0,
        'failed_count': 0,
        'videos': [],
        'documents': []
    }

    for filepath, (docs, video_info) in results.items():
        if docs and video_info:
            formatted_results['success_count'] += 1
            formatted_results['documents'].extend(docs)

            video_data = {
                'filepath': filepath,
                'filename': video_info.filename,
                'filesize': video_info.filesize,
                'duration': video_info.duration,
                'transcript_source': video_info.transcript_source.value if video_info.transcript_source else None,
                'has_transcript': video_info.has_transcript,
                'word_count': len(video_info.transcript_text.split()) if video_info.transcript_text else 0
            }
            formatted_results['videos'].append(video_data)
        else:
            formatted_results['failed_count'] += 1

    return formatted_results
