from moviepy.editor import *
import moviepy.video.fx.all as vfx
import os
import shutil
import openai
import requests
from PIL import Image, ImageFilter
from dotenv import load_dotenv
import random
import tempfile
from pydub import AudioSegment
import soundfile as sf
from google.cloud import speech_v1p1beta1 as speech
import io
import math
import ffmpeg

load_dotenv()

# API Keys
openai.api_key = os.getenv("OPENAI_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
pexels_api_key = os.getenv("PEXELS_API_KEY")
elevenlabs_voice_id = "pqHfZKP75CvOlQylNhV4"

# Ensure output folder exists
def ensure_output_folder(folder="output"):
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder}")
    return folder

# Generate script using OpenAI
def generate_script():
    topics = [
        "craziest historical deaths or assassinations",
        "bizarre medical practices",
        "shocking ancient rituals",
        "wildest historical events",
        "weird historical moments",
    ]
    selected_topic = random.choice(topics)
    prompt = (
        f"Write a gripping 30-second YouTube Short script about {selected_topic}. "
        f"The script should be a fast-paced, attention-grabbing monologue formatted as plain text, "
        f"with no stage directions, music cues, or narrator labels. End with a call-to-action: "
        f"'Like and subscribe for more shocking history stories!' Focus on crazy facts, surprising twists, "
        f"or little-known details to appeal to young adults."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a creative and engaging scriptwriter."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.9,
        )
        script = response['choices'][0]['message']['content'].strip()
        print("Script generated successfully!")
        return script
    except Exception as e:
        print(f"Error generating script: {e}")
        return None

# Generate voiceover using ElevenLabs API
def generate_voiceover(script, output_file):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{elevenlabs_voice_id}"
    headers = {
        "xi-api-key": elevenlabs_api_key,
        "Content-Type": "application/json"
    }
    data = {"text": script, "model_id": "eleven_monolingual_v1"}
    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            with open(output_file, "wb") as f:
                f.write(response.content)
            print(f"Voiceover saved to {output_file}")
        else:
            print(f"Error generating voiceover: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Error generating voiceover: {e}")

# Fetch images from Pexels API based on the topic
def fetch_images(topic, output_folder, count=30):  # Increased to 30 images
    url = f"https://api.pexels.com/v1/search?query={topic}&per_page={count}&orientation=portrait"
    headers = {"Authorization": pexels_api_key}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        images = response.json().get("photos", [])
        for i, img in enumerate(images):
            img_url = img['src']['portrait']
            img_path = os.path.join(output_folder, f"image_{i+1}.jpg")
            img_data = requests.get(img_url).content
            with open(img_path, 'wb') as f:
                f.write(img_data)
            print(f"Downloaded: {img_path}")
    except Exception as e:
        print(f"Error fetching images: {e}")

# Blur an image using Pillow
def blur_image(image_path, output_path, blur_radius=10):
    try:
        image = Image.open(image_path)
        blurred_image = image.filter(ImageFilter.GaussianBlur(blur_radius))
        blurred_image.save(output_path)
        print(f"Blurred image saved to: {output_path}")
    except Exception as e:
        print(f"Error blurring image: {e}")

# Add motion (Ken Burns effect)
def add_motion_to_image(image_path, duration, zoom_start=1.0, zoom_end=1.2):
    clip = ImageClip(image_path, duration=duration)
    zoom_effect = clip.resize(lambda t: zoom_start + (zoom_end - zoom_start) * (t / duration))
    return zoom_effect

# Create video from blurred images and audio with rapid-fire effect
def create_video_from_images_and_audio(images_folder, audio_path, output_video_path, fps=24):
    try:
        images = sorted([os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.endswith((".jpg", ".png"))])
        if not images:
            print("No images found in the folder. Aborting video creation.")
            return

        # Set dimensions for YouTube Shorts
        width, height = 1080, 1920

        audio = AudioFileClip(audio_path)
        duration_per_image = max(audio.duration / len(images), 0.5)

        clips = []
        for img in images:
            # Load and resize image to fill frame properly
            image = ImageClip(img)
            
            # Calculate scaling to fill frame while maintaining aspect ratio
            aspect_ratio = image.w / image.h
            target_aspect = width / height

            if aspect_ratio > target_aspect:
                # Image is wider than frame
                new_height = height
                new_width = int(height * aspect_ratio)
            else:
                # Image is taller than frame
                new_width = width
                new_height = int(width / aspect_ratio)

            # Resize image
            resized_clip = image.resize((new_width, new_height))
            
            # Center crop to target dimensions
            x_center = (resized_clip.w - width) // 2
            y_center = (resized_clip.h - height) // 2
            cropped_clip = resized_clip.crop(
                x1=x_center,
                y1=y_center,
                x2=x_center + width,
                y2=y_center + height
            )

            # Add zoom effect
            clip = cropped_clip.set_duration(duration_per_image)
            clip = clip.resize(lambda t: 1.0 + 0.2 * t/duration_per_image)

            clips.append(clip)

        video = concatenate_videoclips(clips, method="compose")
        video = video.set_audio(audio)

        video.write_videofile(output_video_path, 
                            codec='libx264', 
                            audio_codec='aac',
                            fps=fps,
                            preset='medium',
                            ffmpeg_params=['-pix_fmt', 'yuv420p'])
        print(f"Video created successfully: {output_video_path}")
    except Exception as e:
        print(f"Error creating video: {e}")

# Add captions to video
def add_captions_to_video(video_path, output_path, word_groups):
    try:
        clip = VideoFileClip(video_path)
        width, height = 1080, 1920
        
        text_clips = []
        
        def bounce_effect(t, clip_t, duration):
            if t < clip_t or t > clip_t + duration:
                return 1.0
            progress = (t - clip_t) / duration
            if progress < 0.15:
                return 1.0 + 0.3 * (progress / 0.15)
            elif progress < 0.3:
                return 1.3 - 0.3 * ((progress - 0.15) / 0.15)
            return 1.0 + 0.05 * math.sin(progress * 8 * math.pi)

        for group in word_groups:
            duration = group['end_time'] - group['start_time']
            
            # Calculate vertical position (middle of screen)
            y_pos = height // 2
            
            # Create text clip with size constraint
            txt_clip = (TextClip(group['text'],
                fontsize=120,
                color='yellow',
                stroke_color='black',
                stroke_width=5,
                font='Roboto-Bold.ttf',
                size=(width * 0.9, None),  # Limit width to 90% of frame
                method='caption',
                align='center')
                .set_position(('center', y_pos))  # Fixed position
                .set_start(group['start_time'])
                .set_duration(duration))

            # Add effects
            txt_clip = txt_clip.resize(lambda t: bounce_effect(t, group['start_time'], duration))
            txt_clip = txt_clip.crossfadein(0.05).crossfadeout(0.05)

            # Add shadow with fixed position
            shadow = (TextClip(group['text'],
                fontsize=120,
                color='black',
                stroke_width=0,
                font='Roboto-Bold.ttf',
                size=(width * 0.9, None),
                method='caption',
                align='center')
                .set_position(('center', y_pos + 4))  # Fixed position slightly below
                .set_start(group['start_time'])
                .set_duration(duration)
                .set_opacity(0.5))

            text_clips.extend([shadow, txt_clip])

        video_with_captions = CompositeVideoClip([clip] + text_clips, size=(width, height))

        video_with_captions.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            fps=24,
            preset='medium',
            ffmpeg_params=['-pix_fmt', 'yuv420p']
        )
        
        print(f"Video with captions saved to: {output_path}")
    except Exception as e:
        print(f"Error adding captions to video: {e}")

def speed_up_video(input_path, output_path, speed_factor=1.5):
    try:
        import ffmpeg
        
        # Create temporary files
        temp_fast_video = "temp_fast_video.mp4"
        temp_audio = "temp_audio.wav"
        temp_fast_audio = "temp_fast_audio.wav"
        
        # Extract audio from input video
        stream = ffmpeg.input(input_path)
        stream.audio.output(temp_audio).overwrite_output().run(capture_stdout=True, capture_stderr=True)
        
        # Speed up video (without audio)
        video = VideoFileClip(input_path)
        fast_video = video.without_audio().speedx(speed_factor)
        fast_video.write_videofile(temp_fast_video, 
                                 codec='libx264',
                                 audio=False,
                                 fps=video.fps)
        
        # Speed up audio with pitch correction using ffmpeg
        try:
            stream = ffmpeg.input(temp_audio)
            stream = ffmpeg.filter(stream, 'atempo', str(speed_factor))
            stream = ffmpeg.output(stream, temp_fast_audio)
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
        except ffmpeg.Error as e:
            print('stdout:', e.stdout.decode('utf8'))
            print('stderr:', e.stderr.decode('utf8'))
            raise e
        
        # Combine fast video with pitch-corrected audio
        fast_video = VideoFileClip(temp_fast_video)
        fast_audio = AudioFileClip(temp_fast_audio)
        
        final = fast_video.set_audio(fast_audio)
        
        # Write final video
        final.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            fps=video.fps,
            preset='medium',
            ffmpeg_params=['-pix_fmt', 'yuv420p']
        )
        
        # Clean up
        video.close()
        fast_video.close()
        final.close()
        
        # Remove temporary files
        for temp_file in [temp_fast_video, temp_audio, temp_fast_audio]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        print(f"Video sped up and saved to: {output_path}")
        
    except Exception as e:
        print(f"Error speeding up video: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        for temp_file in [temp_fast_video, temp_audio, temp_fast_audio]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

# Delete all images in the folder
def delete_images_in_folder(folder):
    try:
        shutil.rmtree(folder)
        print(f"Deleted all images in folder: {folder}")
    except Exception as e:
        print(f"Error deleting images: {e}")

# Function to transcribe audio using Google Cloud Speech-to-Text
def transcribe_audio_to_text(audio_path):
    client = speech.SpeechClient()

    with io.open(audio_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz=44100,
        language_code="en-US",
        enable_word_time_offsets=True
    )

    response = client.recognize(config=config, audio=audio)

    # Group into smaller phrases (1-2 words max)
    word_groups = []
    current_group = []
    current_start_time = 0
    
    for result in response.results:
        alternative = result.alternatives[0]
        words = alternative.words
        
        for i, word_info in enumerate(words):
            current_group.append(word_info.word.upper())  # Convert to uppercase immediately
            
            if len(current_group) == 1:
                current_start_time = word_info.start_time.total_seconds()
            
            # Create a group when we have 2 words or at punctuation or it's the last word
            if (len(current_group) == 2 or 
                i == len(words) - 1 or 
                any(punct in word_info.word for punct in [',', '.', '!', '?'])):
                
                word_groups.append({
                    'text': ' '.join(current_group),
                    'start_time': current_start_time,
                    'end_time': word_info.end_time.total_seconds()
                })
                current_group = []

    # Add a small gap between groups for faster pacing
    for i in range(len(word_groups)-1):
        gap = 0.1  # 100ms gap
        if word_groups[i]['end_time'] + gap < word_groups[i+1]['start_time']:
            word_groups[i]['end_time'] = word_groups[i]['end_time'] + gap

    return word_groups

# Comment out main function except for testing speed_up
def main():
    """Test speed up function"""
    input_video = "output/final_video_with_captions.mp4"
    output_video = "output/final_video_sped_up.mp4"
    speed_up_video(input_video, output_video, speed_factor=1.5)

if __name__ == "__main__":
    main()
