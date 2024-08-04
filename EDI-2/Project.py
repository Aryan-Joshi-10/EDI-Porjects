from flask import Flask, render_template_string, request, jsonify
from PIL import Image
import io
import base64
import os
from datetime import datetime
from ultralytics import YOLO

app = Flask(__name__)

# Initialize the YOLO model
yolo_model = YOLO('runs/classify/train5/weights/best.pt')


# Index HTML template string
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Upload Image</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color:#272727; /* Black background */
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            position: relative;
        }
        .container {
            text-align: center;
            background-color: rgba(255, 255, 255, 0.8);  /* White with 80% opacity */
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            z-index: 1;
        }
        h1 {
            color: #333333;
        }
        input[type="file"] {
            padding: 10px;
            margin: 20px 0;
        }
        button[type="submit"] {
            padding: 10px 20px;
            background-color: #5A9;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .bg-text {
            position: absolute;
            top: 20%;
            left: 50%;
            transform: translateX(-50%);
            color: #ffffff; /* Bright white color */
            font-size: 50px;
            font-weight: bold;
            opacity: 0.5; /* Adjust opacity as needed */
            z-index: 0;
        }
    </style>
</head>
<body>
    <div class="bg-text">FAKE IMAGE DETECTION</div>
    <div class="container">
        <h1>Upload Image</h1>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file">
            <br>
            <button type="submit">Predict</button>
        </form>
    </div>
</body>
</html>
"""

# Result HTML template string
RESULT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Prediction Result</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #272727; /* Black background */
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .container {
            text-align: center;
            background-color: rgba(255, 255, 255, 0.8); /* White with 80% opacity */
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            z-index: 1;
            backdrop-filter: blur(5px); /* Apply blur effect */
        }
        h1 {
            color: #333333;
        }
        p {
            margin-bottom: 10px;
        }
        button {
            padding: 10px 20px;
            background-color: #5A9; /* Same color as the Predict button */
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .modal {
            display: none;
            position: fixed;
            z-index: 1;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0,0,0,0.4);
        }
        .modal-content {
            background-color: #fefefe;
            margin: 15% auto;
            padding: 20px;
            border: 1px solid #888;
            width: 80%;
            max-width: 500px;
            border-radius: 10px;
        }
        .toast {
            visibility: hidden;
            min-width: 250px;
            margin-left: -125px;
            background-color: #333;
            color: #fff;
            text-align: center;
            border-radius: 5px;
            padding: 16px;
            position: fixed;
            z-index: 1;
            left: 50%;
            bottom: 30px;
            font-size: 17px;
        }
        .toast.show {
            visibility: visible;
            -webkit-animation: fadein 0.5s, fadeout 0.5s 2.5s;
            animation: fadein 0.5s, fadeout 0.5s 2.5s;
        }
        @-webkit-keyframes fadein {
            from {bottom: 0; opacity: 0;}
            to {bottom: 30px; opacity: 1;}
        }
        @keyframes fadein {
            from {bottom: 0; opacity: 0;}
            to {bottom: 30px; opacity: 1;}
        }
        @-webkit-keyframes fadeout {
            from {bottom: 30px; opacity: 1;}
            to {bottom: 0; opacity: 0;}
        }
        @keyframes fadeout {
            from {bottom: 30px; opacity: 1;}
            to {bottom: 0; opacity: 0;}
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Image Prediction Result</h1>
        <img src="data:image/jpeg;base64,{{ image }}" alt="Uploaded Image" style="max-width: 300px;">
        <h2>Prediction:</h2>
        <p>Real: {{ real_percentage }}%</p>
        <p>Fake: {{ fake_percentage }}%</p>
        <button onclick="openModal()">Report</button>
    </div>

    <!-- Modal -->
    <div id="myModal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeModal()">&times;</span>
            <h2>Report</h2>
            <form action="/report" method="post">
                <label for="query">Enter your query:</label><br>
                <textarea id="query" name="query" rows="4" cols="50"></textarea><br><br>
                <label for="correct_prediction">This should be:</label><br>
                <select id="correct_prediction" name="correct_prediction">
                    <option value="Real">Real</option>
                    <option value="Fake">Fake</option>
                </select><br><br>
                <input type="hidden" id="image_data" name="image_data" value="{{ image }}">
                <button type="submit" style="background-color: #5A9; color: white; border: none; border-radius: 5px; cursor: pointer;">Submit</button>
            </form>
        </div>
    </div>

    <!-- Toast message -->
    <div id="toast" class="toast"></div>

    <script>
        // Open the modal
        function openModal() {
            document.getElementById("myModal").style.display = "block";
        }

        // Close the modal
        function closeModal() {
            document.getElementById("myModal").style.display = "none";
        }

        // Show toast message
        function showToast(message) {
            var toast = document.getElementById("toast");
            toast.innerHTML = message;
            toast.className = "toast show";
            setTimeout(function(){ toast.className = toast.className.replace("show", ""); }, 3000);
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        try:
            # Open the uploaded image using Pillow
            image = Image.open(io.BytesIO(file.read())).convert("RGB")  # Convert to RGB
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            # Perform image prediction using YOLO model
            results = yolo_model(image)

            # Extract predictions
            prediction_str = ""
            real_percentage = 0
            fake_percentage = 0
            for result in results:
                names_dict = result.names
                probs = result.probs.tolist()
                for name, prob in zip(names_dict, probs):
                    # Convert 1 to "Real" and 0 to "Fake"
                    label = "Real" if name == 1 else "Fake"
                    prediction_str += f"{label}: {prob:.2f}<br>"
                    if label == "Real":
                        real_percentage = prob * 100
                    else:
                        fake_percentage = prob * 100

            return render_template_string(RESULT_HTML, image=img_str, real_percentage=real_percentage,
                                          fake_percentage=fake_percentage)
        except Exception as e:
            return str(e)

@app.route('/report', methods=['POST'])
def report():
    query = request.form.get('query')
    correct_prediction = request.form.get('correct_prediction')
    image_data = request.form.get('image_data')

    # Decode the image data
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))

    # Save the image with a timestamp to a folder on the desktop
    save_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'reported_images', correct_prediction)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{correct_prediction}_{timestamp}.jpg"
    full_path = os.path.join(save_path, filename)
    image.save(full_path)

    # Save the query to a text file in the same folder
    query_filename = f"{correct_prediction}_{timestamp}.txt"
    query_full_path = os.path.join(save_path, query_filename)
    with open(query_full_path, 'w') as query_file:
        query_file.write(query)

    # Save the report data (e.g., to a file or database)
    report_data = {
        'query': query,
        'correct_prediction': correct_prediction,
        'image_data': image_data,
        'timestamp': datetime.now().isoformat()
    }

    # Save the report to a text file for simplicity
    if not os.path.exists('reports'):
        os.makedirs('reports')
    report_filename = f'reports/report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    with open(report_filename, 'w') as report_file:
        report_file.write(str(report_data))

    return jsonify({'status': 'success', 'message': 'Report submitted successfully.'})

if __name__ == '__main__':
    app.run(debug=True)