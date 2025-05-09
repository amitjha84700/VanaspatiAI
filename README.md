# VanaspatiAI
Okay, here's a GitHub README description for your Vanaspati AI project. I've tried to capture the essence, key features, and setup instructions.

# Vanaspati AI: Plant Disease Detection & Health Advisor

Vanaspati AI is a Flask-based web application designed to help users identify plant diseases from leaf images. It utilizes a pre-trained TensorFlow/Keras deep learning model for prediction and fetches detailed health information, treatment recommendations, and care steps from a MongoDB database. The application also generates a comprehensive PDF report of the analysis.

## Key Features

*   **Image Upload & Analysis:** User-friendly interface to upload plant leaf images (PNG, JPG, JPEG).
*   **AI-Powered Disease Detection:** Employs a deep learning model (`plant_model.h5`) to classify plant diseases and identify healthy plants.
*   **Detailed Health Information:** Retrieves comprehensive data from MongoDB, including:
    *   Disease description
    *   Recommended treatments/medicines and application procedures
    *   Step-by-step cure and management guidance
    *   Optimal environmental conditions
    *   A summary of the diagnosis and advice
*   **Confidence Score:** Displays the model's confidence in its prediction.
*   **Dynamic Image Display:** Shows the uploaded image alongside the analysis results.
*   **PDF Report Generation:** Creates a professional, downloadable PDF report containing:
    *   User and plant name (if provided)
    *   Analysis timestamp
    *   The uploaded image
    *   Detected plant, disease, and confidence score
    *   All retrieved health information and recommendations
    *   Customizable footer with application details
*   **User-Specific Reports:** Reports can be personalized with a username and plant name entered by the user.
*   **Error Handling & Logging:** Robust error handling and detailed logging for easier debugging.

## Technologies Used

*   **Backend:** Python, Flask
*   **Machine Learning:** TensorFlow, Keras, NumPy
*   **Database:** MongoDB (Atlas) with PyMongo
*   **PDF Generation:** ReportLab
*   **Image Processing:** Pillow (PIL Fork), Matplotlib (for displaying the uploaded image with a title in the web UI)
*   **Frontend:** HTML (rendered via Flask templates), (CSS & JavaScript would be used for enhanced UI, though not explicitly detailed in `app.py`)
*   **Development Server:** Flask Development Server (for local testing)

## Project Structure


.
├── app.py # Main Flask application logic
├── plant_model.h5 # Pre-trained Keras model (NEEDS TO BE ADDED)
├── templates/ # HTML templates
│ ├── index.html # Home page
│ ├── analyze.html # Image upload page
│ ├── result.html # Displays analysis results
│ └── error.html # Generic error page
├── uploads/ # Directory for storing uploaded images (created automatically)
├── reports/ # Directory for storing generated PDF reports (created automatically)
├── static/ # (Optional) For CSS, JS, static images
└── requirements.txt # Python package dependencies (YOU NEED TO CREATE THIS)

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Create a `requirements.txt` file with the following content (you might need to adjust versions based on your setup):
    ```txt
    Flask
    Werkzeug
    tensorflow
    numpy
    matplotlib
    pymongo
    reportlab
    Pillow
    # Add any other specific versions if necessary, e.g., pymongo[srv] for DNS seedlist
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

4.  **MongoDB Setup:**
    *   Create a free MongoDB Atlas account and set up a cluster.
    *   In your cluster, create a database (e.g., `plant_health_db` as specified by `DATABASE_NAME` in `app.py`).
    *   Create a collection within that database (e.g., `plant_health` as specified by `COLLECTION_NAME`).
    *   **Populate the Collection:** Add documents to your `plant_health` collection. Each document should represent a specific plant-disease combination or a healthy plant. The expected structure is implied by `get_plant_health_data()` function:
        ```json
        {
          "plant_name": "Apple", // Underscores in DB, e.g., "Apple"
          "disease_name": "Apple_scab", // Underscores in DB, e.g., "Apple_scab" or "healthy"
          "medicine_name": "Specific Fungicide X",
          "medicine_procedure": "Apply every 2 weeks during wet season...",
          "cure_steps": [
            "Prune infected leaves and branches.",
            "Ensure good air circulation.",
            "Apply recommended fungicide."
          ],
          "disease_description": "Apple scab is a common disease caused by the fungus Venturia inaequalis...",
          "environment": "Favors cool, wet conditions. Ensure proper spacing for air flow.",
          "summary": "Monitor for early signs and apply fungicide preventatively."
        }
        ```
    *   **Configure MongoDB URI:**
        *   **CRITICAL SECURITY WARNING:** The current code hardcodes MongoDB credentials. **DO NOT USE THIS IN PRODUCTION.**
        *   For local development, update `db_user` and `db_password_hardcoded` in `app.py` with your MongoDB Atlas user credentials.
        *   `MONGO_URI = f"mongodb+srv://{db_user}:{db_password_hardcoded}@cluster0.xrtq6na.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"`
        *   Ensure your cluster's network access rules allow connections from your IP address.
        *   **For Production:** Use environment variables or a secrets management system for credentials.

5.  **Place the Model:**
    *   Download or ensure your trained Keras model file is named `plant_model.h5`.
    *   Place `plant_model.h5` in the root directory of the project (same level as `app.py`).

6.  **Verify Model Input Size:**
    *   Open `app.py` and find the constant `MODEL_INPUT_SIZE = (256, 256)`.
    *   **CRITICAL:** This `(height, width)` tuple **MUST** match the input size your `plant_model.h5` expects (excluding the batch and channel dimensions). The script attempts to verify this on startup, but manual confirmation is best. If your model expects `(224, 224)`, change this constant accordingly.

7.  **Verify Class Names Mapping:**
    *   Review the `class_names_mapping` dictionary in `app.py`. The keys (0, 1, 2, ...) must correspond to the output neuron indices of your model, and the values must match the exact class names your model was trained on (e.g., "PlantName___DiseaseName" or "PlantName___healthy"). The script will warn you if the number of classes doesn't match the model's output layer units.

## Running the Application

1.  Ensure your virtual environment is activated and all dependencies are installed.
2.  Navigate to the project's root directory.
3.  Run the Flask application:
    ```bash
    python app.py
    ```
4.  Open your web browser and go to: `http://127.0.0.1:5000/`

## Usage

1.  Navigate to the home page (`/`) or the analysis page (`/analyze`).
2.  Optionally, enter your name and the name/type of the plant you are analyzing.
3.  Click "Choose File" to select an image of a plant leaf.
4.  Click "Analyze Plant".
5.  The application will process the image, display the prediction (plant, disease, confidence), show the uploaded image, and provide detailed health information from the database.
6.  A "Download PDF Report" button will be available to save a detailed report of the analysis.

## Important Considerations

*   **Security:** As mentioned, **never hardcode credentials in production code.** Use environment variables or a secure secrets management solution for your MongoDB URI.
*   **Model Accuracy:** The accuracy of disease detection depends entirely on the quality and training of your `plant_model.h5`.
*   **Database Content:** The usefulness of the health information relies on the comprehensiveness and accuracy of the data in your MongoDB collection.
*   **Production Deployment:** The Flask development server (`app.run(debug=True)`) is not suitable for production. Use a production-grade WSGI server like Gunicorn or Waitress.
*   **Error Logging:** Check the console output where `app.py` is running for detailed logs and error messages.

## Future Enhancements (Potential Ideas)

*   User authentication system.
*   History of user analyses.
*   Batch image processing.
*   API endpoints for programmatic access.
*   More interactive charts or visualizations of prediction certainties.
*   Integration with weather data for environmental context.
*   Admin interface for managing database content.

---

Feel free to fork, improve, and contribute!
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

To make this README fully functional for your GitHub repo:

Create requirements.txt: As detailed in the "Setup" section.

Add your plant_model.h5: Ensure it's in the root directory.

Update <your-repository-url> and <repository-name>: In the "Clone the Repository" step.

Review and customize: Adapt any sections to better fit specifics you might have omitted or to highlight particular aspects.

MongoDB Data: You'll need to populate your MongoDB instance with relevant data for the application to be truly useful. The example JSON in the README provides a schema.
