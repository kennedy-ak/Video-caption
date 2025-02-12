# app.py
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import cv2
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import tempfile
import os
from werkzeug.utils import secure_filename
from transformers import pipeline

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Load the model and processor globally
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
summarizer = pipeline("summarization", model="Falconsai/text_summarization")



def extract_frames(video_path, frame_rate=1):
    print("extracting frame ......")
    """
    Extracts frames from a video at a specified frame rate without saving them.

    :param video_path: Path to the video file.
    :param frame_rate: Number of frames to extract per second.
    :return: List of extracted frames (as numpy arrays).
    """
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get frames per second
    frame_interval = int(fps / frame_rate)  # Interval between frames

    frames = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop if the video ends

        if count % frame_interval == 0:
            frames.append(frame)  # Store frame in list

        count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"frames extrated {len(frame)}")
    return frames


def generate_captions(frames):
    print("generating captions....")
    all_generated_captions = []
    for frame in frames:
        image = Image.fromarray(frame)
        image = image.convert("RGB")
        inputs = processor(images=image,text="the video frame contains", return_tensors="pt")
        captions = model.generate(**inputs)
        final_captions = processor.decode(captions[0], skip_special_tokens=True)
        # print(final_captions)
        all_generated_captions.append(final_captions)

    final_final_captions = " ".join(all_generated_captions)
    print("captions generated")
    return final_final_captions

def generate_summary(text):
    generated_caption = summarizer(text, max_length=30, min_length=20, do_sample=False)
    print(generated_caption[0]['summary_text'])
    return generated_caption[0]['summary_text'] 



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_video', methods=['POST'])
def process_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # extract frame
            frames = extract_frames(filepath)

            # generate captions
            text = generate_captions(frames)


            # get summary
            final_text = generate_summary(text)

            return jsonify({
                'captions': final_text,
                'num_frames_processed': len(frames)
            })

        except Exception as e:
            return jsonify({'error': f'Processing error: {str(e)}'}), 400
        # finally:
        #     # Clean up the uploaded file
        #     if os.path.exists(filepath):
        #         os.remove(filepath)

    except Exception as e:
        return jsonify({'error': f'Upload error: {str(e)}'}), 400
    
if __name__ == '__main__':
    app.run(debug=True)