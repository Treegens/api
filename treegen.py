import tempfile
from flask import Flask, request, jsonify, send_file, make_response, render_template
from ultralytics import YOLO
import cv2
import os
import numpy as np
from io import BytesIO

# Load the YOLO model
model = YOLO('best.pt')
print("Model loaded successfully.")

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/detect_seedlings', methods=['POST'])
def detect_seedlings():
    # Check if a file is present in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    # Read the uploaded file
    uploaded_file = request.files['file']

    # Determine if the file is an image or video based on its extension
    file_extension = os.path.splitext(uploaded_file.filename)[1].lower()
    seedling_count = 0
    output_path = 'test_results/output.mp4'  # Define output path for videos

    if file_extension in ['.jpg', '.jpeg', '.png']:
        # Handle image files
        image_np = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Make predictions on the input image
        results = model.predict(source=img)

        # Count detected seedlings and draw bounding boxes
        for result in results:
            boxes = result.boxes
            seedling_count += len(boxes.data)  # Count the number of boxes
            
            # Draw bounding boxes on the image
            for box in boxes.data:
                if box.shape[0] < 6:  # Ensure there are enough elements in the box
                    continue
                
                x1, y1, x2, y2, conf, cls = box[:6]  # Unpack the bounding box coordinates
                
                # Draw the bounding box (blue color)
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                # Put the class and confidence on the image
                cv2.putText(img, f"Seedling {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Convert the image with bounding boxes back to a file-like object
        _, img_encoded = cv2.imencode('.jpg', img)
        img_io = BytesIO(img_encoded.tobytes())

        # Create a JSON response with the image and seedling count
        response = {
            'seedling_count': seedling_count,
            'image': img_io.getvalue().decode('latin-1')  # Convert bytes to string for JSON response
        }
        return jsonify(response)

    elif file_extension in ['.mp4', '.avi', '.mov']:
        # Handle video files
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_video_file:
            uploaded_file.save(temp_video_file.name)  # Save the video to disk

            video_capture = cv2.VideoCapture(temp_video_file.name)
            frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(video_capture.get(cv2.CAP_PROP_FPS))
            
            # Define the codec and create VideoWriter object to save output video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
            out_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

            # Iterate over each frame in the video
            while video_capture.isOpened():
                ret, frame = video_capture.read()
                if not ret:
                    break

                # Make predictions on the frame
                results = model.predict(source=frame)

                # Count detected seedlings and draw bounding boxes
                for result in results:
                    boxes = result.boxes
                    seedling_count += len(boxes.data)  # Count the number of boxes
                    
                    # Draw bounding boxes on the frame
                    for box in boxes.data:
                        if box.shape[0] < 6:  # Ensure there are enough elements in the box
                            continue
                        
                        x1, y1, x2, y2, conf, cls = box[:6]  # Unpack the bounding box coordinates
                        
                        # Draw the bounding box (blue color)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        # Put the class and confidence on the frame
                        cv2.putText(frame, f"Seedling {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Write the processed frame to the output video
                out_video.write(frame)

            # Release resources
            video_capture.release()
            out_video.release()

        # Create a response with the output video and seedling count
        with open(output_path, 'rb') as video_file:
            video_data = video_file.read()
        response = {
            'seedling_count': seedling_count,
            'video': video_data.decode('latin-1')  # Convert bytes to string for JSON response
        }
        return jsonify(response)

    else:
        return jsonify({'error': 'Unsupported file type. Please upload an image or video.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
