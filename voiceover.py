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
        "gory scary murder story",
        "real horror story",
        "serial killer story",
    ]
    selected_topic = random.choice(topics)
    prompt = (
        f"Write a gripping 30-second YouTube Short script about a REAL {selected_topic}. "
        f"Include specific details like:\n"
        f"- Exact dates and locations\n"
        f"- Names of real victims\n"
        f"- Graphic details of how the crime happened\n"
        f"- Shocking evidence found\n"
        f"- Unexpected twists in the investigation\n"
        f"The script should be extremely fast-paced and intense, starting with a hook like "
        f"'In 1984, Sarah Johnson thought it was just another normal night until...' "
        f"Pack in as many specific, horrifying details as possible in a rapid-fire style. "
        f"Focus on the most shocking and gruesome aspects of the true story. "
        f"End with 'Like and subscribe for more shocking true crime stories!' "
        f"Format as plain text with no stage directions. Make it feel urgent and terrifying."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a true crime storyteller who specializes in shocking, detailed accounts of real events."},
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
def fetch_images(topic, output_folder, count=15):
    """Get random images from the local image bank"""
    try:
        image_bank_path = "image_bank"
        categories = [
            'gore',
            'shadows',
            'faces',
            'dark_scenes',
            'crime_scenes',
            'creepy_dolls',
            'vintage_horror'
        ]
        
        # Weight categories based on relevance (can be adjusted)
        category_weights = {
            'gore': 3,
            'shadows': 2,
            'faces': 2,
            'dark_scenes': 2,
            'crime_scenes': 3,
            'creepy_dolls': 1,
            'vintage_horror': 2
        }
        
        selected_images = []
        
        # Get weighted random selection of categories
        weighted_categories = []
        for cat, weight in category_weights.items():
            weighted_categories.extend([cat] * weight)
        
        # Select images from categories
        while len(selected_images) < count:
            category = random.choice(weighted_categories)
            category_path = os.path.join(image_bank_path, category)
            
            if os.path.exists(category_path):
                available_images = [f for f in os.listdir(category_path) 
                                  if f.endswith(('.jpg', '.png'))]
                
                if available_images:
                    img = random.choice(available_images)
                    source_path = os.path.join(category_path, img)
                    dest_path = os.path.join(output_folder, f"image_{len(selected_images)+1}.jpg")
                    
                    # Copy image to output folder
                    shutil.copy2(source_path, dest_path)
                    selected_images.append(dest_path)
                    print(f"Selected: {img} from {category}")
        
        print(f"Selected {len(selected_images)} images from image bank")
        return True
        
    except Exception as e:
        print(f"Error selecting images from bank: {e}")
        return False

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

        # Load main voiceover and background audio
        voiceover = AudioFileClip(audio_path)
        background = AudioFileClip("bg.mp3").subclip(15)  # Start at 15 seconds
        
        # Trim background to match voiceover length
        background = background.subclip(0, voiceover.duration)
        background = background.volumex(0.15)  # 15% volume
        
        # Combine audio tracks
        final_audio = CompositeAudioClip([voiceover, background])

        # Rest of video creation (unchanged)
        width, height = 1080, 1920
        duration_per_image = max(voiceover.duration / len(images), 0.5)

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
        video = video.set_audio(final_audio)  # Using combined audio

        video.write_videofile(output_video_path, 
                            codec='libx264', 
                            audio_codec='aac',
                            fps=fps,
                            preset='medium',
                            ffmpeg_params=['-pix_fmt', 'yuv420p'])
        
        print(f"Video created successfully with background audio: {output_video_path}")
        
    except Exception as e:
        print(f"Error creating video: {e}")
        import traceback
        traceback.print_exc()

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

        # Define colors to alternate between
        colors = ['yellow', 'white']
        current_color_index = 0
        
        for group in word_groups:
            duration = group['end_time'] - group['start_time']
            
            # Get current color and switch for next word
            current_color = colors[current_color_index]
            current_color_index = (current_color_index + 1) % len(colors)
            
            # Calculate vertical position (middle of screen)
            y_pos = height // 2
            
            # Create text clip with alternating colors
            txt_clip = (TextClip(group['text'],
                fontsize=120,
                color=current_color,
                stroke_color='black',
                stroke_width=8,
                font='ObelixProIt-cyr.ttf',
                size=(width * 0.9, None),
                method='caption',
                align='center')
                .set_position(('center', y_pos))
                .set_start(group['start_time'])
                .set_duration(duration))

            # Add effects
            txt_clip = txt_clip.resize(lambda t: bounce_effect(t, group['start_time'], duration))
            txt_clip = txt_clip.crossfadein(0.05).crossfadeout(0.05)

            text_clips.append(txt_clip)

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

def speed_up_video(input_path, output_path, speed_factor=1.23):
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

def determine_topic(script):
    prompt = (
        f"Given this script: \"{script}\", determine the main topic in one or two words. "
        "It should relate broadly to the content but doesn't have to match the exact event. "
        "For example, for Apollo 11, use 'space'."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=20,
            temperature=0.7,
        )
        topic = response['choices'][0]['message']['content'].strip()
        print(f"Identified Topic: {topic}")
        return topic
    except Exception as e:
        print(f"Error determining topic: {e}")
        return None

# Comment out main function except for testing speed_up
def main():
    try:
        # Create output folder
        output_folder = ensure_output_folder()
        images_folder = ensure_output_folder("images")

        # Step 1: Generate script
        print("\nGenerating script...")
        script = generate_script()
        if not script:
            return
        print(f"\nGenerated Script:\n{script}\n")

        # Step 2: Determine main topic for image search
        print("\nDetermining main topic for images...")
        search_topic = determine_topic(script)
        if not search_topic:
            search_topic = " ".join(script.split()[0:3])  # Fallback to first 3 words
        print(f"Using search topic: {search_topic}")

        # Step 3: Generate voiceover
        print("\nGenerating voiceover...")
        voiceover_file = os.path.join(output_folder, "voiceover.mp3")
        # generate_voiceover(script, voiceover_file)

        # Step 4: Transcribe the voiceover to text with timing
        print("\nTranscribing voiceover to text...")
        word_groups = transcribe_audio_to_text(voiceover_file)
        print("\nTranscription with timing:")
        for group in word_groups:
            print(f"{group['text']}: {group['start_time']:.2f}s - {group['end_time']:.2f}s")

        # Step 5: Fetch and process images using determined topic
        print(f"\nFetching images for topic: {search_topic}...")
        fetch_images(search_topic, images_folder)

        # Step 6: Create base video
        print("\nCreating video from images and audio...")
        base_video = os.path.join(output_folder, "base_video.mp4")
        create_video_from_images_and_audio(images_folder, voiceover_file, base_video)

        # Step 7: Add captions
        print("\nAdding captions to video...")
        captioned_video = os.path.join(output_folder, "final_video_with_captions.mp4")
        add_captions_to_video(base_video, captioned_video, word_groups)

        # Step 8: Speed up the video
        print("\nSpeeding up video...")
        final_video = os.path.join(output_folder, "final_video_sped_up.mp4")
        speed_up_video(captioned_video, final_video, speed_factor=1.2)

        # Clean up images
        delete_images_in_folder(images_folder)

        print("\nVideo generation complete!")
        print(f"Final video saved to: {final_video}")

    except Exception as e:
        print(f"Error in main process: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
