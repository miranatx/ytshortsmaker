from moviepy.editor import *
import moviepy.video.fx.all as vfx
import os
import shutil
import openai
import requests
from PIL import Image, ImageFilter
from dotenv import load_dotenv
import random

load_dotenv()

# API Keys
#OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# ElevenLabs API Key
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

# Pexels API Key
pexels_api_key = os.getenv("PEXELS_API_KEY")

elevenlabs_voice_id = "pqHfZKP75CvOlQylNhV4"  # Replace with a valid voice ID


# Ensure output folder exists
def ensure_output_folder(folder="output"):
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder}")
    return folder

# Generate script using OpenAI
def generate_script():
    topics = [
        "craziest historical deaths or assasinations",
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

# Determine the main topic of the script using OpenAI
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

# Fetch images from Pexels API based on the topic
def fetch_images(topic, output_folder, count=10):
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

# Create video from blurred images and audio with motion
def create_video_from_images_and_audio(images_folder, audio_path, output_video_path, fps=24):
    try:
        images = sorted([os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.endswith((".jpg", ".png"))])
        if not images:
            print("No images found in the folder. Aborting video creation.")
            return

        audio = AudioFileClip(audio_path)
        duration_per_image = audio.duration / len(images)

        clips = [
            add_motion_to_image(
                img, duration=duration_per_image, zoom_start=1.0, zoom_end=1.2
            ) for img in images
        ]
        video = concatenate_videoclips(clips, method="compose")
        video = video.set_audio(audio)

        video.write_videofile(output_video_path, codec="libx264", audio_codec="aac", fps=fps)
        print(f"Video created successfully: {output_video_path}")
    except Exception as e:
        print(f"Error creating video: {e}")

# Speed up the video by a factor
def speed_up_video(input_path, output_path, speed_factor=1.25):
    try:
        clip = VideoFileClip(input_path)
        sped_up_clip = clip.fx(vfx.speedx, speed_factor)
        sped_up_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        print(f"Sped-up video saved to: {output_path}")
    except Exception as e:
        print(f"Error speeding up video: {e}")

# Delete all images in the folder
def delete_images_in_folder(folder):
    try:
        shutil.rmtree(folder)
        print(f"Deleted all images in folder: {folder}")
    except Exception as e:
        print(f"Error deleting images: {e}")

# Main function to run the pipeline
def main():
    output_folder = ensure_output_folder()
    images_folder = ensure_output_folder("images")

    # Step 1: Generate the script
    script = generate_script()
    if script is None:
        return

    print("\nGenerated Script:")
    print(script)

    # Step 2: Generate the voiceover
    voiceover_file = os.path.join(output_folder, "voiceover.mp3")
    print("\nGenerating voiceover...")
    generate_voiceover(script, voiceover_file)

    # Step 3: Determine topic
    topic = determine_topic(script)
    if not topic:
        return

    # Step 4: Fetch images
    print("\nFetching images...")
    fetch_images(topic, images_folder, count=10)

    # Step 5: Blur images
    print("\nBlurring images...")
    blurred_folder = ensure_output_folder("blurred_images")
    for img in os.listdir(images_folder):
        if img.endswith((".jpg", ".png")):
            blur_image(
                os.path.join(images_folder, img),
                os.path.join(blurred_folder, img),
                blur_radius=6
            )

    # Step 6: Create video from blurred images
    final_video_path = os.path.join(output_folder, "final_video.mp4")
    print("\nCreating video...")
    create_video_from_images_and_audio(blurred_folder, voiceover_file, final_video_path)

    # Step 7: Speed up the video
    sped_up_video_path = os.path.join(output_folder, "final_video_sped_up.mp4")
    print("\nSpeeding up video...")
    speed_up_video(final_video_path, sped_up_video_path, speed_factor=1.25)

    # Step 8: Clean up images
    print("\nDeleting images...")
    delete_images_in_folder(images_folder)
    delete_images_in_folder(blurred_folder)

if __name__ == "__main__":
    main()
