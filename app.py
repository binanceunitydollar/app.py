from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import os

app = Flask(__name__)

# Define a simple generator model (same as before)
class Generator(nn.Module):
    # ... (Same code as before) ...

@app.route('/generate-video', methods=['POST'])
def generate_video():
    try:
        input_text = request.form.get('inputText')
        text_vector = torch.randn(10)
        generator = Generator()
        generated_image = generator(text_vector)
        generated_image_np = (generated_image.view(28, 28) * 255).byte().cpu().numpy()

        # Create a temporary video file
        video_path = 'temp_video.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 30
        frame_size = (28, 28)
        out = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

        for i in range(150):  # Creating 150 frames (5 seconds of video)
            frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
            frame[:, :, 0] = generated_image_np
            out.write(frame)

        out.release()

        return jsonify({'videoPath': video_path})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
