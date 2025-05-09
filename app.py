import os
import io
import base64
import traceback # For detailed error logging
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_file, session, flash
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib
matplotlib.use('Agg') # Use Agg backend for Matplotlib in Flask to avoid GUI issues
import matplotlib.pyplot as plt
from pymongo import MongoClient
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportlabImage, Frame, PageTemplate, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT # For text alignment
from reportlab.lib.units import inch
from reportlab.lib import colors # Import colors
from PIL import Image as PILImage
from datetime import datetime

# --- Constants and Configurations ---
UPLOAD_FOLDER = 'uploads'
REPORTS_FOLDER = "reports"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# --- MongoDB Configuration ---
# !!! WARNING: Hardcoding credentials like this is insecure and NOT RECOMMENDED for production! !!!
# This password is now visible directly in the code. Consider using environment variables
# or a secrets management system (like HashiCorp Vault, AWS Secrets Manager, etc.).
db_user = "modi"
db_password_hardcoded = "modi" # <-- YOUR PASSWORD IS HARDCODED HERE - VERY INSECURE

# Use f-string with the hardcoded password
MONGO_URI = f"mongodb+srv://{db_user}:{db_password_hardcoded}@cluster0.xrtq6na.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

DATABASE_NAME = "plant_health_db" # IMPORTANT: Verify this is your database name on Atlas
COLLECTION_NAME = "plant_health" # IMPORTANT: Verify this is your collection name

# Footer Configuration for PDF
APP_NAME = "Vanaspati AI"
FOOTER_SERVICE_POINTS = [
    "- Accurate Plant Disease Detection",
    "- Detailed Health Information & Care Steps",
    "- Environment & Nutrient Recommendations" # Updated service point
]
FOOTER_QUOTE = '"The glory of gardening: hands in the dirt, head in the sun, heart with nature. To nurture a garden is to feed not just the body, but the soul." - Alfred Austin'

# Get the absolute path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Create necessary directories with absolute paths if they don't exist
os.makedirs(os.path.join(current_dir, UPLOAD_FOLDER), exist_ok=True)
os.makedirs(os.path.join(current_dir, REPORTS_FOLDER), exist_ok=True)

# --- !!! IMPORTANT: VERIFY THIS VALUE !!! ---
# This MUST match the input size your TensorFlow model expects.
# Common sizes are (224, 224), (256, 256), (299, 299), etc.
# Check your model's documentation or use model.summary() after loading.
MODEL_INPUT_SIZE = (256, 256) # <-- EXAMPLE VALUE - CHANGE IF YOUR MODEL NEEDS DIFFERENT SIZE

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(current_dir, UPLOAD_FOLDER)

# --- !!! IMPORTANT: Change this secret key for production environments !!! ---
# Use a strong, random key. You can generate one using: python -c 'import os; print(os.urandom(24))'
app.secret_key = 'dev_secret_key_change_for_prod' # CHANGE THIS!

# --- Custom Filter for Base64 Encoding ---
@app.template_filter('b64encode')
def b64encode_filter(data):
    """Encodes bytes or BytesIO object to base64 string for HTML embedding."""
    if isinstance(data, io.BytesIO):
        data.seek(0)
        return base64.b64encode(data.read()).decode('utf-8')
    elif isinstance(data, bytes):
         return base64.b64encode(data).decode('utf-8')
    try:
        # Fallback for other types, attempt to encode as string
        return base64.b64encode(str(data).encode('utf-8')).decode('utf-8')
    except Exception:
        app.logger.warning(f"b64encode filter failed for data type: {type(data)}", exc_info=True)
        return ""

# --- Load Model and Mapping ---
MODEL_PATH = os.path.join(current_dir, "plant_model.h5") # Ensure your model file is in the same directory
model = None
model_input_shape_verified = None # To store the expected shape like (None, H, W, C)

try:
    print(f"Attempting to load model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
         raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please place 'plant_model.h5' in the same directory as the script.")
    model = tf.keras.models.load_model(MODEL_PATH)
    model_input_shape_verified = model.input_shape
    print("-" * 60)
    print(f"Model loaded successfully from: {MODEL_PATH}")
    print(f"Expected Input Shape by loaded model: {model_input_shape_verified}")
    print("Model Summary:")
    model.summary(print_fn=lambda x: print(x)) # Print summary to console
    print("-" * 60)

    # Verify if loaded model input shape's H, W matches the constant
    if isinstance(model_input_shape_verified, tuple) and len(model_input_shape_verified) >= 3:
        loaded_h, loaded_w = model_input_shape_verified[1:3] # Typically (None, H, W, C)
        if (loaded_h, loaded_w) != MODEL_INPUT_SIZE:
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"CRITICAL WARNING: Model Input Size Mismatch!")
            print(f"Configured MODEL_INPUT_SIZE {MODEL_INPUT_SIZE} does NOT match the")
            print(f"loaded model's expected input dimensions ({loaded_h}, {loaded_w}).")
            print(f"--> Please UPDATE the MODEL_INPUT_SIZE constant in the script")
            print(f"    to ({loaded_h}, {loaded_w}) for accurate predictions. <--")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # You might want to exit here in a production scenario if the mismatch is critical
            # exit(1)
    else:
        print(f"WARNING: Could not automatically verify model input dimensions H, W from shape {model_input_shape_verified}.")
        print(f"Please manually check if it aligns with MODEL_INPUT_SIZE {MODEL_INPUT_SIZE}.")

except FileNotFoundError as fnf_error:
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"FATAL ERROR: {fnf_error}")
    print(f"Application cannot proceed without the model file.")
    print(f"Ensure 'plant_model.h5' exists at: {MODEL_PATH}")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    exit(1) # Stop the application
except Exception as e:
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"FATAL ERROR: An unexpected error occurred loading model from {MODEL_PATH}:")
    print(f"{e}")
    print(f"Application cannot proceed without a functional model.")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    traceback.print_exc() # Print detailed traceback
    exit(1) # Stop the application


# --- Class Names Mapping ---
# MAKE SURE THIS ORDER AND CONTENT MATCHES YOUR MODEL'S OUTPUT CLASSES EXACTLY
# The index (0, 1, 2, ...) must correspond to the output neuron index for that class.
class_names_mapping = {
    0: "Apple___Apple_scab", 1: "Apple___Black_rot", 2: "Apple___Cedar_apple_rust", 3: "Apple___healthy",
    4: "Blueberry___healthy", 5: "Cherry_(including_sour)___Powdery_mildew", 6: "Cherry_(including_sour)___healthy",
    7: "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", 8: "Corn_(maize)___Common_rust_", 9: "Corn_(maize)___Northern_Leaf_Blight",
    10: "Corn_(maize)___healthy", 11: "Grape___Black_rot", 12: "Grape___Esca_(Black_Measles)", 13: "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    14: "Grape___healthy", 15: "Orange___Haunglongbing_(Citrus_greening)", 16: "Peach___Bacterial_spot", 17: "Peach___healthy",
    18: "Pepper,_bell___Bacterial_spot", 19: "Pepper,_bell___healthy", 20: "Potato___Early_blight", 21: "Potato___Late_blight",
    22: "Potato___healthy", 23: "Raspberry___healthy", 24: "Soybean___healthy", 25: "Squash___Powdery_mildew",
    26: "Strawberry___Leaf_scorch", 27: "Strawberry___healthy", 28: "Tomato___Bacterial_spot", 29: "Tomato___Early_blight",
    30: "Tomato___Late_blight", 31: "Tomato___Leaf_Mold", 32: "Tomato___Septoria_leaf_spot",
    33: "Tomato___Spider_mites Two-spotted_spider_mite", 34: "Tomato___Target_Spot", 35: "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    36: "Tomato___Tomato_mosaic_virus", 37: "Tomato___healthy"
}

# Verify class mapping length matches model output units
if model and hasattr(model, 'output_shape'):
    try:
        # Handle potential nested tuples/lists in output_shape
        shape = model.output_shape
        if isinstance(shape, (list, tuple)) and len(shape) > 0:
            # If multiple outputs, check the last one (common for classification)
            # or adjust if your model structure is different
            last_output_shape = shape[-1] if isinstance(shape[-1], (list, tuple)) else shape
            if isinstance(last_output_shape, (list, tuple)) and len(last_output_shape) > 0:
                 model_output_units = last_output_shape[-1]
                 if isinstance(model_output_units, int) and model_output_units != len(class_names_mapping):
                      print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                      print(f"WARNING: Model output layer has {model_output_units} units, but class_names_mapping has {len(class_names_mapping)} entries.")
                      print(f"WARNING: Predictions might map to incorrect names. Please verify your class_names_mapping dictionary.")
                      print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                 elif not isinstance(model_output_units, int):
                      print(f"WARNING: Could not determine numeric output units from model output shape: {shape}")
            else:
                 print(f"WARNING: Could not determine output units from model output shape structure: {shape}")
        else:
            print(f"WARNING: Unexpected model output shape format: {shape}")
    except Exception as ex:
         print(f"WARNING: Error checking model output units against class mapping: {ex}")


# --- Helper Functions ---

def allowed_file(filename):
    """Checks if the filename has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path, target_size=MODEL_INPUT_SIZE):
    """Loads and preprocesses an image for model prediction."""
    try:
        app.logger.debug(f"Preprocessing image: {img_path} to target size: {target_size}")
        # Load image using tf.keras.preprocessing
        img = image.load_img(img_path, target_size=target_size)
        # Convert image to numpy array
        img_array = image.img_to_array(img)
        # Expand dimensions to create a batch (even if it's just one image)
        img_array = np.expand_dims(img_array, axis=0)
        # Normalize pixel values to [0, 1] (common practice)
        img_array /= 255.0
        app.logger.debug(f"Preprocessed image shape: {img_array.shape}")
        return img_array
    except FileNotFoundError:
        app.logger.error(f"Image file not found at {img_path}")
        raise # Re-raise to be caught by the route handler
    except Exception as e:
        app.logger.error(f"Could not load or process image {img_path}: {e}", exc_info=True)
        # Raise a more specific error to the user
        raise ValueError(f"Failed to preprocess image: {os.path.basename(img_path)}. It might be corrupted or an unsupported format.") from e

def get_plant_disease_name(predicted_class_index, mapping):
    """Gets plant and disease names from the predicted class index using the mapping."""
    class_name = mapping.get(predicted_class_index)
    if class_name:
        parts = class_name.split("___", 1)
        plant = parts[0].replace("_", " ") # Replace underscores for display
        disease = "Healthy" # Default
        if len(parts) == 2 and parts[1].lower() != "healthy":
            disease = parts[1].replace("_", " ") # Replace underscores
        elif len(parts) == 1 and "healthy" not in plant.lower():
             # Handle cases where maybe only plant name is given and it's not healthy
             disease = "Condition Unknown/Check"
             app.logger.warning(f"Class name '{class_name}' for index {predicted_class_index} only had one part and wasn't 'healthy'. Setting disease to '{disease}'.")
        elif len(parts) == 2 and parts[1].lower() == "healthy":
            disease = "Healthy" # Explicitly set if ___healthy is used

        app.logger.debug(f"Mapped index {predicted_class_index} ('{class_name}') to Plant: '{plant}', Disease: '{disease}'")
        return plant, disease
    else:
        app.logger.warning(f"Predicted class index {predicted_class_index} not found in class_names_mapping.")
        return "Unknown Plant", "Unknown Condition"


def connect_to_mongodb():
    """Connects to MongoDB Atlas using the HARCODED MONGO_URI and returns the collection object."""
    try:
        app.logger.info(f"Attempting to connect to MongoDB Atlas (using hardcoded credentials - INSECURE)...")
        # Increased timeout for potentially slow connections
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=15000)
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ping')
        app.logger.info("Connected to MongoDB Atlas successfully!")
        db = client[DATABASE_NAME]
        return db[COLLECTION_NAME]
    except Exception as e:
        app.logger.error(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", exc_info=True)
        app.logger.error(f"ERROR: Could not connect to MongoDB Atlas: {e}")
        app.logger.error(f"Check network configuration, Atlas IP Whitelist, and the hardcoded credentials in the script.")
        app.logger.error(f"Mongo URI used (password redacted): {MONGO_URI.replace(db_password_hardcoded, '***REDACTED***')}")
        app.logger.error(f"Database: {DATABASE_NAME}, Collection: {COLLECTION_NAME}")
        app.logger.error(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return None # Return None to indicate connection failure

def get_plant_health_data(plant_name, disease_name):
    """Retrieves plant health data (including nutrient info) from MongoDB."""
    collection = connect_to_mongodb()

    # Define default structure, now including nutrient_info
    default_data = {
        "plant_name": plant_name,
        "disease_name": disease_name,
        "medicine_name": "N/A",
        "medicine_procedure": "N/A",
        "cure_steps": ["N/A"],
        "disease_description": "Details not available.",
        "environment": "N/A",
        "summary": "Information not found in database.",
        "nutrient_info": None # Default nutrient info to None
    }

    if collection is None:
        app.logger.warning("MongoDB connection failed. Cannot retrieve plant health data.")
        default_data["summary"] = "Database connection failed. Cannot retrieve details."
        return default_data

    try:
        # Prepare query terms (replace spaces with underscores, handle 'healthy' case)
        query_plant = plant_name.strip().replace(" ", "_")
        # Ensure consistent 'healthy' check
        if disease_name.strip().lower() == "healthy":
            query_disease = "healthy"
        else:
            query_disease = disease_name.strip().replace(" ", "_")

        query = {"plant_name": query_plant, "disease_name": query_disease}
        app.logger.debug(f"Querying MongoDB with: {query}")
        plant_data_from_db = collection.find_one(query)

        if plant_data_from_db:
            app.logger.info(f"Data found in MongoDB for {plant_name} - {disease_name}.")
            # Start with defaults and update with DB data to ensure all keys exist
            result_data = default_data.copy()
            result_data["plant_name"] = plant_name # Use the display name
            result_data["disease_name"] = disease_name # Use the display name

            # Safely update fields from DB, keeping defaults if DB field is missing/null
            result_data["medicine_name"] = plant_data_from_db.get("medicine_name") or "N/A"
            result_data["medicine_procedure"] = plant_data_from_db.get("medicine_procedure") or "N/A"
            result_data["disease_description"] = plant_data_from_db.get("disease_description") or "No description available."
            result_data["environment"] = plant_data_from_db.get("environment") or "No specific environmental notes."
            result_data["summary"] = plant_data_from_db.get("summary") or "No summary provided."

             # *** NEW: Get nutrient_info ***
            db_nutrient_info = plant_data_from_db.get("nutrient_info")
            if isinstance(db_nutrient_info, dict):
                 # Basic validation: Ensure required keys have some content if they exist
                valid_nutrient_info = {}
                req_nut = db_nutrient_info.get('required_nutrients')
                def_imp = db_nutrient_info.get('deficiency_impact')
                perc_guide = db_nutrient_info.get('percentage_guidelines')

                if req_nut and str(req_nut).strip(): valid_nutrient_info['required_nutrients'] = str(req_nut)
                if def_imp and str(def_imp).strip(): valid_nutrient_info['deficiency_impact'] = str(def_imp)
                if perc_guide and str(perc_guide).strip(): valid_nutrient_info['percentage_guidelines'] = str(perc_guide)

                if valid_nutrient_info: # Only assign if we found some valid info
                     result_data["nutrient_info"] = valid_nutrient_info
                     app.logger.debug("Nutrient info processed from DB.")
                else:
                     app.logger.debug("Nutrient info field existed in DB but contained no valid data.")
                     result_data["nutrient_info"] = None # Ensure it's None if empty/invalid
            else:
                 app.logger.debug("Nutrient info not found or not a dictionary in DB document.")
                 result_data["nutrient_info"] = None # Ensure it's None if missing

            # Special handling for 'cure_steps' which should be a list of strings
            db_steps = plant_data_from_db.get('cure_steps')
            if isinstance(db_steps, list):
                # Filter out empty strings or None values, convert others to string
                result_data['cure_steps'] = [str(item).strip() for item in db_steps if item and str(item).strip()] or ["N/A"]
            elif isinstance(db_steps, str) and db_steps.strip() and db_steps != 'N/A':
                 # If it's a single string, put it in a list
                result_data['cure_steps'] = [db_steps.strip()]
            else:
                result_data['cure_steps'] = ["N/A"] # Default if not a list or valid string

            # Ensure other text fields are strings
            for key in ["medicine_name", "medicine_procedure", "disease_description", "environment", "summary"]:
                 if not isinstance(result_data[key], str):
                      result_data[key] = str(result_data.get(key, 'N/A'))

            # Optional: Include the MongoDB document ID if needed later
            if "_id" in plant_data_from_db:
                result_data["_id"] = str(plant_data_from_db["_id"])

            return result_data
        else:
            app.logger.warning(f"No specific data found for {plant_name} ({query_plant}) - {disease_name} ({query_disease}) in the database.")
            # Provide specific info for 'healthy' case even if not explicitly in DB
            if disease_name.lower() == "healthy":
                default_data["disease_description"] = "The plant appears to be healthy based on the visual analysis."
                default_data["cure_steps"] = [
                    "Maintain consistent and appropriate watering.",
                    "Ensure adequate sunlight or suitable light conditions.",
                    "Provide balanced nutrients/fertilizer if required for the species.",
                    "Monitor regularly for any signs of pests or stress.",
                    "Ensure good air circulation."
                ]
                default_data["summary"] = "Plant diagnosed as healthy. Continue standard care and monitoring."
                default_data["medicine_name"] = "Not Applicable"
                default_data["medicine_procedure"] = "Not Applicable"
                # Attempt to get 'healthy' nutrient info if it exists for the plant
                healthy_query = {"plant_name": query_plant, "disease_name": "healthy"}
                healthy_data_from_db = collection.find_one(healthy_query)
                if healthy_data_from_db and isinstance(healthy_data_from_db.get("nutrient_info"), dict):
                    db_nutrient_info = healthy_data_from_db.get("nutrient_info")
                    valid_nutrient_info = {}
                    req_nut = db_nutrient_info.get('required_nutrients')
                    def_imp = db_nutrient_info.get('deficiency_impact') # Often less relevant for healthy
                    perc_guide = db_nutrient_info.get('percentage_guidelines')
                    if req_nut and str(req_nut).strip(): valid_nutrient_info['required_nutrients'] = str(req_nut)
                    if def_imp and str(def_imp).strip(): valid_nutrient_info['deficiency_impact'] = str(def_imp)
                    if perc_guide and str(perc_guide).strip(): valid_nutrient_info['percentage_guidelines'] = str(perc_guide)
                    if valid_nutrient_info:
                         default_data["nutrient_info"] = valid_nutrient_info
                         app.logger.debug("Added general healthy nutrient info for this plant type.")

            else:
                 # Keep the default "Information not found" for non-healthy cases not in DB
                 default_data["summary"] = f"No specific management information found for '{disease_name}' on '{plant_name}' in the database."
                 default_data["nutrient_info"] = None # Ensure nutrient info is None

            return default_data
    except Exception as e:
        app.logger.error(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", exc_info=True)
        app.logger.error(f"ERROR: Error during MongoDB query or data processing: {e}")
        # Log the query safely (might contain PII if plant/disease names are sensitive, but generally ok)
        app.logger.error(f"Query attempted: {{'plant_name': '{query_plant}', 'disease_name': '{query_disease}'}}") # Log query structure
        app.logger.error(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # Return default data with an error message in summary
        default_data["summary"] = "A database query error occurred while retrieving details. Please check server logs."
        default_data["nutrient_info"] = None # Ensure nutrient info is None on error
        return default_data


def create_plot_base64(image_path, plant_name, disease_name, confidence):
    """Creates a plot *of the image* with prediction title and returns base64 encoded string."""
    try:
        app.logger.debug(f"Creating plot for {os.path.basename(image_path)}")
        plt.figure(figsize=(8, 6)) # Adjust size as needed

        # Use PIL to open the image for plotting to handle different types gracefully
        with PILImage.open(image_path) as img:
            # Ensure image is in RGB mode for matplotlib compatibility
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            elif img.mode == 'P': # Palette mode
                img = img.convert('RGBA').convert('RGB') # Convert via RGBA
            elif img.mode == 'LA': # Luminance + Alpha
                 img = img.convert('RGB')

            plt.imshow(np.array(img)) # Display the actual image

        # Create a title with prediction details
        title = f"Prediction: {plant_name} - {disease_name}\nConfidence: {confidence:.2f}%"
        plt.title(title, fontsize=12, wrap=True)
        plt.axis("off") # Hide axes

        # Save the plot to a BytesIO buffer
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close() # Close the figure to free memory
        img_bytes.seek(0) # Rewind the buffer

        app.logger.debug(f"Plot created successfully.")
        # Encode the plot image bytes to base64
        return b64encode_filter(img_bytes)
    except FileNotFoundError:
        app.logger.error(f"ERROR creating plot: Image file not found at {image_path}")
        return None
    except Exception as e:
        app.logger.error(f"ERROR creating plot: {e}", exc_info=True)
        return None


def footer_canvas(canvas, doc):
    """Draws the footer on each page for the PDF report."""
    try:
        canvas.saveState()
        canvas.setFont('Helvetica', 9)

        # Footer positioning
        footer_y = doc.bottomMargin * 0.5 # Position slightly above the bottom margin
        line_y = footer_y + 15 # Line slightly above text
        page_width = doc.pagesize[0]
        left_margin = doc.leftMargin
        right_margin = doc.rightMargin
        content_width = page_width - left_margin - right_margin

        # Draw horizontal line
        canvas.setStrokeColor(colors.lightgrey)
        canvas.setLineWidth(0.5)
        canvas.line(left_margin, line_y, page_width - right_margin, line_y)

        # Column 1: App Name and Service Points
        col1_x = left_margin
        textobject_col1 = canvas.beginText(col1_x, footer_y)
        textobject_col1.setFont('Helvetica-Bold', 10)
        textobject_col1.setFillColor(colors.HexColor("#1a5f7a")) # Darker blue/teal
        textobject_col1.textLine(f"{APP_NAME}")
        textobject_col1.moveCursor(0, -3) # Small gap
        textobject_col1.setFont('Helvetica', 7.5)
        textobject_col1.setFillColor(colors.darkslategray)
        for point in FOOTER_SERVICE_POINTS:
             textobject_col1.textLine(point)
        canvas.drawText(textobject_col1)

        # Column 2: Quote (aligned right, wrapped)
        col2_x = page_width - right_margin # Right edge for alignment
        canvas.setFont('Helvetica-Oblique', 7.5)
        canvas.setFillColor(colors.dimgray)

        max_line_width_quote = content_width * 0.45 # Allow quote to take ~45% of width
        lines = []
        current_line = ""
        words = FOOTER_QUOTE.split()
        for word in words:
            test_line = f"{current_line} {word}".strip()
            # Check width using the correct font and size
            if canvas.stringWidth(test_line, 'Helvetica-Oblique', 7.5) <= max_line_width_quote:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        if current_line: # Append the last line
            lines.append(current_line)

        # Draw lines from bottom up for right alignment
        line_height = 9 # Points
        start_y_quote = footer_y + (len(lines) -1) * line_height # Adjust start based on number of lines
        current_y_quote = start_y_quote
        for line in reversed(lines): # Draw from top down
             canvas.drawRightString(col2_x, current_y_quote, line)
             current_y_quote -= line_height # Move down for next line

        # Page Number (centered at the very bottom)
        page_num_text = f"Page {canvas.getPageNumber()}"
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.grey)
        canvas.drawCentredString(page_width / 2.0, doc.bottomMargin * 0.20, page_num_text) # Lower position

        canvas.restoreState()
    except Exception as e:
        app.logger.error(f"ERROR drawing PDF footer: {e}", exc_info=True)
        # Ensure canvas state is restored even if error occurs
        if canvas._code: # Check if there's anything to restore
             canvas.restoreState()


def generate_pdf_report(image_path, plant_name, disease_name, confidence, plant_health_data, username, name):
    """Generates a PDF report with improved styling, nutrient table, and footer."""
    now = datetime.now()
    # Sanitize username and name for filename, default if empty
    safe_username = secure_filename(username if username else "user")
    safe_name = secure_filename(name if name else "plant") # Use 'plant' if name is missing
    report_base_filename = f"VanaspatiAI_{safe_username}_{safe_name}_{now.strftime('%Y%m%d_%H%M%S')}.pdf"
    report_disk_path = os.path.join(current_dir, REPORTS_FOLDER, report_base_filename)

    buffer = io.BytesIO() # Create PDF in memory

    # Define styles
    styles = getSampleStyleSheet()
    # Base style
    base_style = ParagraphStyle(name='Base', fontName='Helvetica', fontSize=10, leading=14, spaceAfter=6)
    # Headings
    styleH1 = ParagraphStyle(name='Heading1', parent=base_style, fontName='Helvetica-Bold', fontSize=18, alignment=TA_CENTER, spaceAfter=20, textColor=colors.HexColor("#2c3e50")) # Dark Blue-Gray
    styleH2 = ParagraphStyle(name='Heading2', parent=base_style, fontName='Helvetica-Bold', fontSize=13, spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#34495e"), borderPadding=2, borderBottomWidth=0.5, borderBottomColor=colors.lightgrey) # Underlined H2
    styleH3 = ParagraphStyle(name='Heading3', parent=base_style, fontName='Helvetica-Bold', fontSize=11, spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#1a5f7a")) # Teal Sub-heading
    # Body text
    styleN = ParagraphStyle(name='Normal', parent=base_style, alignment=TA_LEFT, spaceAfter=4)
    styleB = ParagraphStyle(name='BoldLabel', parent=styleN, fontName='Helvetica-Bold', textColor=colors.HexColor("#1a5f7a")) # Teal color for labels
    styleI = ParagraphStyle(name='ItalicNote', parent=styleN, fontName='Helvetica-Oblique', textColor=colors.dimgray, fontSize=9)
    # List items
    styleLi = ParagraphStyle(name='ListItem', parent=styleN, leftIndent=18, spaceBefore=0, spaceAfter=2, bulletIndent=5, firstLineIndent=0)
    # Captions
    styleCaption = ParagraphStyle(name='Caption', parent=styleN, fontSize=9, alignment=TA_CENTER, textColor=colors.grey, spaceBefore=2, spaceAfter=12)
    # Content from DB (used in tables and general text)
    styleDBContent = ParagraphStyle(name='DBContent', parent=styleN, spaceAfter=8) # More space after DB content paragraphs

    story = [] # ReportLab story list

    # 1. Title
    story.append(Paragraph("Plant Health Analysis Report", styleH1))
    story.append(Spacer(1, 0.1 * inch))

    # 2. Report Info Table
    info_data = [
        [Paragraph(f"<b>Report For:</b> {name} ({username})", styleN)],
        [Paragraph(f"<b>Generated On:</b> {now.strftime('%Y-%m-%d %H:%M:%S')}", styleN)],
    ]
    info_table = Table(info_data, colWidths=[6.5*inch]) # Use full width minus margins approx
    info_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 0), # No extra padding for full width feel
        ('LINEBELOW', (0,-1), (-1,-1), 0.5, colors.lightgrey), # Light bottom line
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.2 * inch))

    # 3. Image Analysis Section Title
    story.append(Paragraph("Image Analysis Result", styleH2))

    # 4. Uploaded Image
    img_max_width = 4.0 * inch # Max width for the image in PDF
    img_max_height = 3.5 * inch # Max height
    try:
        app.logger.debug(f"Trying to add image to PDF: {image_path}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at path: {image_path}")

        # Use PIL to get dimensions and check format
        with PILImage.open(image_path) as img_pil:
            # Ensure it's RGB for ReportLab compatibility if needed
            if img_pil.mode in ('RGBA', 'LA', 'P'):
                 app.logger.debug(f"Converting image from {img_pil.mode} to RGB for PDF.")
                 img_pil = img_pil.convert('RGB')
            img_width_px, img_height_px = img_pil.size

        # Calculate aspect ratio and determine PDF image size
        aspect = img_height_px / float(img_width_px) if img_width_px > 0 else 1
        pdf_img_width = img_max_width
        pdf_img_height = pdf_img_width * aspect

        # Adjust if height exceeds max height
        if pdf_img_height > img_max_height:
            pdf_img_height = img_max_height
            pdf_img_width = pdf_img_height / aspect

        # Add the image to the story
        img_for_pdf = ReportlabImage(image_path, width=pdf_img_width, height=pdf_img_height)
        img_for_pdf.hAlign = 'CENTER' # Center the image
        story.append(img_for_pdf)
        # Add caption below image
        story.append(Paragraph(f"<i>Uploaded Image: {os.path.basename(image_path)}</i>", styleCaption))
        app.logger.debug(f"Image added to PDF story with dimensions {pdf_img_width/inch:.2f}\" x {pdf_img_height/inch:.2f}\".")

    except FileNotFoundError as fnf_err:
         error_msg = f"ERROR: Could not find uploaded image file '{os.path.basename(image_path)}' to include in PDF."
         story.append(Paragraph(f"<i>{error_msg}</i>", styleI))
         app.logger.error(error_msg, exc_info=True)
    except Exception as img_err:
        error_msg = f"ERROR including image in PDF: {img_err}"
        story.append(Paragraph(f"<i>{error_msg}</i>", styleI))
        app.logger.error(error_msg, exc_info=True)

    # 5. Prediction Details Table
    pred_data = [
        [Paragraph("Detected Plant:", styleB), Paragraph(plant_name, styleN)],
        [Paragraph("Detected Condition:", styleB), Paragraph(disease_name, styleN)],
        [Paragraph("Confidence Score:", styleB), Paragraph(f"{confidence:.2f}%", styleN)],
    ]
    pred_table = Table(pred_data, colWidths=[1.7*inch, 4.8*inch]) # Adjusted widths
    pred_table.setStyle(TableStyle([
       ('VALIGN', (0,0), (-1,-1), 'TOP'),
       ('BOTTOMPADDING', (0,0), (-1,-1), 4),
       ('TOPPADDING', (0,0), (-1,-1), 0),
       ('LEFTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(pred_table)
    story.append(Spacer(1, 0.2 * inch))

    # 6. Health Info Section Title
    story.append(Paragraph("Health Information & Recommendations", styleH2))

    # 7. Health Information Details (from MongoDB data)
    # Check if we have valid data (not just defaults or error messages)
    has_valid_db_data = (plant_health_data and
                         plant_health_data.get('summary') not in [
                            "Information not found in database.",
                            "Database connection failed. Cannot retrieve details.",
                            "A database query error occurred while retrieving details. Please check server logs."
                         ] and
                         # Also check if it's not the generic 'healthy' summary generated locally
                         not (disease_name.lower() == "healthy" and
                              plant_health_data.get('summary') == "Plant diagnosed as healthy. Continue standard care and monitoring.")
                        )

    # Extract nutrient info for potential use
    nutrient_info = plant_health_data.get('nutrient_info') if plant_health_data else None
    has_valid_nutrient_info = isinstance(nutrient_info, dict) and any(nutrient_info.get(k) for k in ['required_nutrients', 'deficiency_impact', 'percentage_guidelines'])


    if has_valid_db_data or disease_name.lower() == "healthy":
        app.logger.debug("Found valid health data or 'healthy' diagnosis, adding details to PDF.")

        # Helper to add sections cleanly, handling None or 'N/A'
        def add_section(title, content, style=styleDBContent, is_list=False, bold_title=True):
            # Check if content is meaningful (not None, not 'N/A', not empty list/string)
            is_valid_content = False
            if is_list:
                if isinstance(content, list) and any(item and str(item).strip() and str(item) != 'N/A' for item in content):
                    # Filter the list to only contain valid items
                    content = [str(item).strip() for item in content if item and str(item).strip() and str(item) != 'N/A']
                    if content: is_valid_content = True
            elif isinstance(content, str):
                if content.strip() and content != 'N/A':
                    is_valid_content = True
            elif content is not None and not isinstance(content, str): # Handle non-string, non-list if needed
                 content_str = str(content).strip()
                 if content_str and content_str != 'N/A':
                      content = content_str # Use the string representation
                      is_valid_content = True

            # Add to story only if valid content exists
            if is_valid_content:
                title_style = styleB if bold_title else styleN # Choose title style
                story.append(Paragraph(f"{title}:", title_style)) # Use chosen style for title
                if is_list:
                    for item in content:
                        # Basic escaping for safety, replace newlines for ReportLab
                        escaped_item = item.replace('<', '<').replace('>', '>') # HTML escape basic tags
                        story.append(Paragraph(f"• {escaped_item.replace(chr(10), '<br/>').replace('\n', '<br/>')}", styleLi, bulletText='•'))
                else:
                    escaped_content = content.replace('<', '<').replace('>', '>')
                    # Replace newlines with ReportLab's <br/> tag
                    story.append(Paragraph(escaped_content.replace(chr(10), '<br/>').replace('\n', '<br/>'), style))
                story.append(Spacer(1, 0.08 * inch)) # Space after each section
            # else: app.logger.debug(f"Skipping PDF section '{title}' due to invalid/empty content: {content}")


        # Add sections using the helper
        add_section("Disease Description", plant_health_data.get('disease_description'))
        # Only show medicine/procedure if not healthy
        if disease_name.lower() != "healthy":
            add_section("Recommended Treatment / Medicine", plant_health_data.get('medicine_name'))
            add_section("Application Procedure", plant_health_data.get('medicine_procedure'))
        add_section("Management & Care Steps", plant_health_data.get('cure_steps'), is_list=True) # This is expected to be a list
        add_section("Environmental Considerations", plant_health_data.get('environment'))

        # --- Add Nutrient Information Section ---
        if has_valid_nutrient_info:
            story.append(Paragraph("Nutrient Information", styleH3)) # Sub-heading for nutrients
            story.append(Spacer(1, 0.05 * inch))

            # Prepare data for the table
            nutrient_data_table_rows = []

            # Helper to safely get nutrient text and format it
            def get_nutrient_text(key):
                text = nutrient_info.get(key, 'N/A')
                if text and text != 'N/A':
                    # Basic escaping and newline handling for paragraphs
                    escaped_text = str(text).replace('<', '<').replace('>', '>')
                    return escaped_text.replace(chr(10), '<br/>').replace('\n', '<br/>')
                return None # Return None if empty or N/A

            # Required Nutrients
            req_nut_text = get_nutrient_text('required_nutrients')
            if req_nut_text:
                 nutrient_data_table_rows.append([
                     Paragraph("Required Nutrients:", styleB), # Label column
                     Paragraph(req_nut_text, styleDBContent)   # Content column
                 ])

            # Deficiency Impact
            def_impact_text = get_nutrient_text('deficiency_impact')
            if def_impact_text:
                 nutrient_data_table_rows.append([
                     Paragraph("Potential Impact of Imbalance:", styleB),
                     Paragraph(def_impact_text, styleDBContent)
                 ])

            # Percentage Guidelines
            perc_guide_text = get_nutrient_text('percentage_guidelines')
            if perc_guide_text:
                 nutrient_data_table_rows.append([
                     Paragraph("General Guidelines:", styleB),
                     Paragraph(perc_guide_text, styleDBContent)
                 ])

            # Create and style the table only if there are rows to display
            if nutrient_data_table_rows:
                nutrient_table = Table(nutrient_data_table_rows, colWidths=[1.7*inch, 4.8*inch]) # Adjust widths
                nutrient_table.setStyle(TableStyle([
                    ('VALIGN', (0,0), (-1,-1), 'TOP'),         # Align text to top of cell
                    ('LEFTPADDING', (0,0), (-1,-1), 0),       # No extra left padding
                    ('RIGHTPADDING', (0,0), (-1,-1), 5),      # Small right padding for content
                    ('TOPPADDING', (0,0), (-1,-1), 2),        # Minimal top padding
                    ('BOTTOMPADDING', (0,0), (-1,-1), 6),     # Space below text in cell
                    # Optional: Add subtle background for label column
                    # ('BACKGROUND', (0,0), (0,-1), colors.Color(red=(240/255), green=(240/255), blue=(240/255))), # Very light grey
                    # ('LINEBELOW', (0,0), (-1,-1), 0.25, colors.lightgrey), # Optional fine lines between rows
                ]))
                story.append(nutrient_table)
                story.append(Spacer(1, 0.15 * inch)) # Space after the nutrient table
            else:
                 app.logger.debug("No valid nutrient data rows to add to the PDF table.")
        else:
             app.logger.debug(f"Nutrient info not found or invalid for PDF report: {plant_name} - {disease_name}")


        # Add Summary last
        add_section("Summary / Key Advice", plant_health_data.get('summary'))

    else:
        # If no valid DB data AND not healthy, show the summary message (which indicates lack of data)
        not_found_summary = plant_health_data.get('summary', "No detailed health information available.") if plant_health_data else "Health data could not be retrieved."
        story.append(Paragraph(not_found_summary, styleI)) # Use italic style for this note
        if plant_health_data:
             story.append(Spacer(1, 0.1 * inch))
             # Add the search terms for context if data retrieval was attempted
             story.append(Paragraph(f"<i>(Details searched for: Plant='{plant_health_data.get('plant_name', plant_name)}', Condition='{plant_health_data.get('disease_name', disease_name)}')</i>", styleI))
        app.logger.warning(f"No valid health data found or DB error occurred for PDF. Displaying summary: {not_found_summary}")

    # Build the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=1.2*inch) # Increased bottom margin for footer space

    try:
        app.logger.info(f"Building PDF report story...")
        doc.build(story, onFirstPage=footer_canvas, onLaterPages=footer_canvas)
        app.logger.info(f"PDF report generated successfully in memory ({buffer.getbuffer().nbytes} bytes).")

        # Save to disk (optional, but good for persistence/debugging)
        try:
            with open(report_disk_path, 'wb') as f:
                f.write(buffer.getvalue())
            app.logger.info(f"PDF report saved to disk: {report_disk_path}")
        except IOError as io_err:
             app.logger.error(f"Error saving PDF report to disk at {report_disk_path}: {io_err}")

        buffer.seek(0) # Rewind buffer for sending
        return buffer, report_base_filename
    except Exception as e:
        app.logger.error(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", exc_info=True)
        app.logger.error(f"ERROR building PDF report: {e}")
        app.logger.error(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return None, None # Indicate failure


# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the home page."""
    app.logger.info("Serving route: / (index)")
    return render_template('index.html')

@app.route('/analyze', methods=['GET'])
def analyze():
    """Renders the image upload page."""
    app.logger.info("Serving route: /analyze (GET)")
    # Clear any previous PDF filename from session when going back to analyze page
    session.pop('pdf_filename', None)
    return render_template('analyze.html') # Pass error via flash message system instead

@app.route('/result', methods=['POST'])
def result():
    """Handles file upload, prediction, data retrieval, and renders the result page."""
    start_time = datetime.now()
    app.logger.info(f"\n--- Request received for /result (POST) at {start_time} ---")

    if model is None:
         app.logger.critical("Model is not loaded. Cannot perform analysis.")
         flash("Critical Error: The analysis model is not available. Please contact support.", "error")
         return redirect(url_for('analyze'))

    # --- 1. File Handling ---
    if 'file' not in request.files:
        app.logger.warning("No file part in request")
        flash("No file part in the request. Please select a file.", "warning")
        return redirect(url_for('analyze'))

    file = request.files['file']
    username = request.form.get('username', 'User').strip()
    name = request.form.get('name', 'Plant').strip() # Default to 'Plant' if name missing

    if file.filename == '':
        app.logger.warning("No selected file (empty filename)")
        flash("No file selected. Please choose an image file.", "warning")
        return redirect(url_for('analyze'))

    if not allowed_file(file.filename):
        app.logger.warning(f"File type not allowed: {file.filename}")
        ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'N/A'
        flash(f"Invalid file type (.{ext}). Please upload a PNG, JPG, or JPEG image.", "error")
        return redirect(url_for('analyze'))

    filepath = None # Initialize filepath to None
    filename = None
    try:
        # Secure filename and create full path
        filename = secure_filename(file.filename)
        # Add timestamp/unique ID to filename to prevent overwrites if needed
        # timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # unique_filename = f"{timestamp}_{filename}"
        # filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename) # Using original filename for simplicity now

        app.logger.info(f"Saving file '{filename}' from user '{username}' (name: '{name}') to '{filepath}'")
        file.save(filepath)
        app.logger.info(f"File saved successfully.")

        # --- 2. Image Preprocessing ---
        app.logger.info(f"Preprocessing image: {filepath}")
        # Use the globally verified MODEL_INPUT_SIZE
        processed_image = preprocess_image(filepath, target_size=MODEL_INPUT_SIZE)
        app.logger.info(f"Image preprocessed. Shape for model: {processed_image.shape}")

        # --- 3. Model Prediction ---
        app.logger.info(f"Making prediction with model...")
        if model is None: # Double check model is loaded
             raise RuntimeError("Model became unavailable after initial load.")

        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100) # Ensure it's a float
        app.logger.info(f"Prediction done. Class index: {predicted_class_index}, Confidence: {confidence:.2f}%")

        # --- 4. Get Plant/Disease Name ---
        app.logger.info(f"Getting plant/disease name for index {predicted_class_index}")
        plant_name, disease_name = get_plant_disease_name(predicted_class_index, class_names_mapping)
        app.logger.info(f"Identified as: Plant='{plant_name}', Disease='{disease_name}'")

        # --- 5. Retrieve Data from DB ---
        app.logger.info(f"Retrieving health data for '{plant_name}' - '{disease_name}'")
        plant_health_data = get_plant_health_data(plant_name, disease_name) # Includes nutrient info now
        app.logger.info(f"Health data retrieval complete. Summary: {plant_health_data.get('summary', 'N/A')}")
        app.logger.debug(f"Nutrient info retrieved: {'Yes' if plant_health_data.get('nutrient_info') else 'No'}")


        # --- 6. Generate Plot (Optional, e.g., for showing image with title) ---
        app.logger.info(f"Creating plot...")
        plot_base64 = create_plot_base64(filepath, plant_name, disease_name, confidence)
        if plot_base64 is None: app.logger.warning("Plot generation failed.")
        else: app.logger.debug(f"Plot created successfully (Base64 length: {len(plot_base64)}).")

        # --- 7. Generate PDF Report ---
        app.logger.info(f"Generating PDF report...")
        pdf_buffer, pdf_filename = generate_pdf_report(
            filepath, plant_name, disease_name, confidence, plant_health_data, username, name
        )
        pdf_generation_failed = False
        if pdf_buffer and pdf_filename:
             session['pdf_filename'] = pdf_filename # Store filename in session for download link
             app.logger.info(f"PDF generated, filename for download: {pdf_filename}")
        else:
             session.pop('pdf_filename', None) # Ensure no filename if generation failed
             pdf_generation_failed = True
             app.logger.warning("PDF generation failed.")
             # Optionally flash a message to the user about PDF failure
             flash("Could not generate the downloadable PDF report for this analysis.", "warning")


        # --- 8. Prepare Data for Template ---
        # Generate URL for the original uploaded image to display it directly
        uploaded_image_url = None
        if filename:
             try:
                  # Use the potentially unique filename if implemented above
                  # display_filename = unique_filename if 'unique_filename' in locals() else filename
                  display_filename = filename # Using original filename for now
                  uploaded_image_url = url_for('uploaded_file', filename=display_filename, _external=False) # Use relative URL
                  app.logger.debug(f"Generated URL for uploaded image: {uploaded_image_url}")
             except Exception as url_err:
                  app.logger.error(f"Could not generate URL for uploaded file '{filename}': {url_err}", exc_info=True)

        result_data = {
            "plant_name": plant_name,
            "disease_name": disease_name,
            "confidence": confidence,
            "plant_health_data": plant_health_data, # Contains all DB info including nutrients
            "plot_base64": plot_base64, # Base64 string of the plot (image with title)
            "uploaded_image_url": uploaded_image_url, # URL to the raw uploaded image
            "pdf_available": bool(pdf_filename),
            "pdf_generation_failed": pdf_generation_failed,
            "username": username, # Pass username and name back if needed in template
            "name": name,
        }

        end_time = datetime.now()
        processing_time = end_time - start_time
        app.logger.info(f"Result processing successful. Total time: {processing_time}")
        app.logger.info(f"--- Rendering result page for {filename} ---")
        return render_template('result.html', **result_data)

    # --- Error Handling ---
    except ValueError as ve: # Specific error from preprocess_image or potentially data handling
         error_msg = f"Data Processing Error: {ve}."
         app.logger.error(error_msg, exc_info=True)
         flash(f"Failed to process the image: {ve}. It might be corrupted or in an unsupported format.", "error")
         # Clean up uploaded file if it exists
         if filepath and os.path.exists(filepath):
              try: os.remove(filepath); app.logger.info(f"Removed partially uploaded/processed file: {filepath}")
              except OSError as ose: app.logger.warning(f"Could not remove file {filepath} after error: {ose}")
         return redirect(url_for('analyze')) # Redirect back to upload page

    except tf.errors.InvalidArgumentError as tf_err: # Error during model.predict (e.g., shape mismatch)
         error_msg = f"TensorFlow Model Input Error: {tf_err}. This often indicates a mismatch between the image processing dimensions ({MODEL_INPUT_SIZE}) and the model's expected input shape ({model_input_shape_verified})."
         app.logger.error(error_msg, exc_info=True)
         app.logger.error(f"Model expected shape: {model_input_shape_verified}, Configured input size: {MODEL_INPUT_SIZE}")
         flash("Model Error: Could not process the image due to an internal model incompatibility. Please check server logs.", "error")
         # Clean up uploaded file
         if filepath and os.path.exists(filepath):
              try: os.remove(filepath); app.logger.info(f"Removed file: {filepath}")
              except OSError as ose: app.logger.warning(f"Could not remove file {filepath} after TF error: {ose}")
         return redirect(url_for('analyze'))

    except FileNotFoundError as fnf_err: # Error if file disappears between save and process
        error_msg = f"File Not Found Error during processing: {fnf_err}."
        app.logger.error(error_msg, exc_info=True)
        flash("File handling error occurred after upload. Please try again.", "error")
        return redirect(url_for('analyze'))

    except RuntimeError as rte: # General runtime errors (like model becoming None)
        error_msg = f"Runtime Error during analysis: {rte}"
        app.logger.error(error_msg, exc_info=True)
        flash("A runtime error occurred during analysis. Please try again or contact support.", "error")
        if filepath and os.path.exists(filepath):
             try: os.remove(filepath); app.logger.info(f"Removed file: {filepath}")
             except OSError as ose: app.logger.warning(f"Could not remove file {filepath} after runtime error: {ose}")
        return redirect(url_for('analyze'))

    except Exception as e: # Catch-all for any other unexpected errors
        error_msg = f"An unexpected error occurred during analysis: {str(e)}"
        app.logger.error(f"UNEXPECTED ERROR in /result: {error_msg}", exc_info=True) # Log full traceback
        flash("An unexpected internal error occurred. Please try again later or contact support.", "error")
        # Clean up uploaded file if it exists and path is known
        if filepath and os.path.exists(filepath):
             try: os.remove(filepath); app.logger.info(f"Removed file due to unexpected error: {filepath}")
             except OSError as ose: app.logger.warning(f"Could not remove file {filepath} after unexpected error: {ose}")
        return redirect(url_for('analyze'))


@app.route('/download-pdf')
def download_pdf():
    """Serves the generated PDF report for download."""
    pdf_filename = session.get('pdf_filename')
    if not pdf_filename:
        app.logger.warning("PDF download requested, but no PDF filename found in session.")
        flash("Could not find the PDF report to download. It might have expired or failed to generate.", "error")
        # Redirect to index or analyze page, as the result page might not be relevant anymore
        return redirect(url_for('index'))

    # Sanitize filename again just in case
    safe_pdf_filename = secure_filename(pdf_filename)
    pdf_path = os.path.join(current_dir, REPORTS_FOLDER, safe_pdf_filename)
    app.logger.info(f"PDF download request for '{safe_pdf_filename}'. Attempting to send file from: {pdf_path}")

    if os.path.exists(pdf_path):
        try:
            # Use send_file for robust file sending
            return send_file(pdf_path,
                             mimetype='application/pdf',
                             as_attachment=True, # Suggest download
                             download_name=safe_pdf_filename) # Set the filename for the user
        except Exception as e:
            app.logger.error(f"ERROR sending PDF file {pdf_path}: {e}", exc_info=True)
            flash("Server error: Could not send the PDF file.", "error")
            return redirect(url_for('index')) # Redirect on send error
    else:
        app.logger.error(f"ERROR: PDF file not found at path for download: {pdf_path}")
        # Remove the invalid filename from session
        session.pop('pdf_filename', None)
        flash("Error: The generated PDF report file was not found on the server.", "error")
        return redirect(url_for('index'))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serves uploaded images directly."""
    # Security: Use secure_filename to prevent directory traversal attacks
    safe_filename = secure_filename(filename)
    if safe_filename != filename:
         app.logger.warning(f"Invalid filename requested in /uploads/: {filename}. Secured as: {safe_filename}")
         # Return 404 or 400 - 404 is common for resource not found
         return "Invalid filename", 404

    upload_folder_path = app.config['UPLOAD_FOLDER']
    file_path = os.path.join(upload_folder_path, safe_filename)

    # Check if the file actually exists and is a file (not a directory)
    if not os.path.isfile(file_path):
        app.logger.error(f"Requested uploaded file not found or is not a file: {file_path}")
        return "File not found", 404

    try:
        app.logger.debug(f"Serving uploaded file: {file_path}")
        # send_file handles Content-Type and other headers automatically
        return send_file(file_path)
    except Exception as e:
        app.logger.error(f"ERROR serving uploaded file {safe_filename}: {e}", exc_info=True)
        return "Error serving file", 500

@app.route('/error') # Kept for potential explicit redirects, but flash messages are preferred
def error():
    """Displays a generic error page."""
    # This route is less used now with flash messages, but can be a fallback
    error_message = request.args.get('error', 'An unknown error occurred.')
    app.logger.info(f"Serving generic error page with message: {error_message}")
    return render_template('error.html', error=error_message)


# --- Main Execution ---
if __name__ == '__main__':
    print("=" * 60)
    print("Starting Vanaspati AI - Plant Health Analysis Flask App...")
    print(f"Flask App Root Path: {current_dir}")
    print(f"Uploads folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Reports folder: {os.path.join(current_dir, REPORTS_FOLDER)}")
    print(f"Allowed image extensions: {ALLOWED_EXTENSIONS}")
    print(f"Configured Model Input Size (H, W): {MODEL_INPUT_SIZE} (Verify against model summary!)")
    if model:
        print(f"Model Status: Loaded Successfully from {MODEL_PATH}")
        print(f"Model Expected Input Shape: {model_input_shape_verified}")
    else:
         # This case should ideally not be reached due to exit() in load block
        print(f"Model Status: !!! ERROR - Model FAILED to load from {MODEL_PATH} !!!")

    # Updated MongoDB log message - REMEMBER THIS IS INSECURE
    print("-" * 60)
    print("MongoDB Configuration:")
    print("  URI: Connecting to Atlas cluster (credentials HARCODED in script - !!! INSECURE !!!)")
    print(f"  Database: {DATABASE_NAME}")
    print(f"  Collection: {COLLECTION_NAME}")

    print("-" * 60)
    # Check Flask secret key status
    secret_key_status = "!!! USING DEFAULT DEV KEY - CHANGE THIS FOR PRODUCTION !!!"
    if app.secret_key != 'dev_secret_key_change_for_prod':
         secret_key_status = "*** CUSTOM KEY SET (Recommended for Production) ***"
    print(f"Flask Secret Key Status: {secret_key_status}")
    print(f"Debug Mode: {app.debug}") # Check if debug mode is on
    print("-" * 60)
    print("Starting Flask development server...")
    print("Access the app at http://127.0.0.1:5000/ (or your configured host/port)")
    print("Press CTRL+C to quit.")
    print("=" * 60)

    # Use Flask's development server. For production, use a WSGI server like Gunicorn or Waitress.
    # debug=True enables auto-reloading and detailed error pages (DO NOT use in production)
    app.run(debug=True, host='0.0.0.0', port=5000) # host='0.0.0.0' makes it accessible on your network