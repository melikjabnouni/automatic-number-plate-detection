import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
from datetime import datetime
import easyocr

# Configuration
INPUT_SOURCE = r"C:\Users\melik\Desktop\projet\testimgv.jpg"
MODEL_PATH = r"C:\Users\melik\Desktop\projet\utils\best.onnx"  # Absolute path to model
CROPPED_PLATES_DIR = r"C:\Users\melik\Desktop\projet\cropped_plates"
OUTPUT_LOG = r"C:\Users\melik\Desktop\projet\plate_log.txt"
CONFIDENCE_THRESHOLD = 0.5
# Government plate prefixes and their corresponding ministries
GOVERNMENT_PLATES = {
    "01": "رئاسة الحكومة",
    "02": "وزارة الداخلية",
    "03": "وزارة العدل",
    "04": "وزارة الشؤون الخارجية",
    "05": "وزارة الدفاع الوطني",
    "06": "وزارة الشؤون الدينية",
    "07": "وزارة التنمية و الإستثمار",
    "08": "وزارة المالية",
    "09": "وزارة الصناعة",
    "10": "وزارة التجارة",
    "11": "وزارة الفلاحة",
    "12": "وزارة أملاك الدولة",
    "13": "وزارة التجهيز",
    "14": "وزارة البيئة",
    "15": "وزارة النقل",
    "16": "وزارة السياحة",
    "17": "وزارة التكنولوجيا والاتصالات",
    "18": "وزارة التعليم العالي و التربية",
    "19": "وزارة الثقافة",
    "20": "وزارة الصحة",
    "21": "وزارة الشؤون الاجتماعية",
    "22": "وزارة التشغيل والتكوين المهني",
    "23": "وزارة الشباب والرياضة"
}

# Ensure cropped plates directory exists
os.makedirs(CROPPED_PLATES_DIR, exist_ok=True)

# Initialize YOLOv8
print(f"Loading YOLO model from {MODEL_PATH}")
model = YOLO(MODEL_PATH, task="detect")
print("YOLO model loaded successfully.")

# Initialize EasyOCR reader with Arabic and English support
reader = easyocr.Reader(['ar', 'en'], gpu=False)  # Set gpu=True if you have a GPU

def enhance_image(image):
    if image is None:
        raise ValueError("Image not found or failed to load!")
    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Upscale image
    upscale_factor = 6
    resized = cv2.resize(image, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC)
    return resized

# Simple Centroid Tracker
class CentroidTracker:
    def __init__(self, max_age=30, max_distance=30):
        self.next_id = 1
        self.tracks = {}
        self.max_age = max_age
        self.max_distance = max_distance

    def update(self, boxes):
        centroids = []
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            centroids.append((cx, cy, box))

        if not self.tracks:
            for centroid in centroids:
                self.tracks[self.next_id] = {
                    "centroid": (centroid[0], centroid[1]),
                    "box": centroid[2],
                    "age": 0
                }
                self.next_id += 1
            return list(self.tracks.items())

        updated_tracks = {}
        for track_id, track in self.tracks.items():
            track["age"] += 1
            if track["age"] > self.max_age:
                continue

            min_dist = float("inf")
            best_centroid = None
            for centroid in centroids:
                cx, cy = centroid[0], centroid[1]
                dist = np.sqrt((cx - track["centroid"][0]) ** 2 + (cy - track["centroid"][1]) ** 2)
                if dist < min_dist and dist < self.max_distance:
                    min_dist = dist
                    best_centroid = centroid

            if best_centroid:
                updated_tracks[track_id] = {
                    "centroid": (best_centroid[0], best_centroid[1]),
                    "box": best_centroid[2],
                    "age": 0
                }
                centroids.remove(best_centroid)

        for centroid in centroids:
            updated_tracks[self.next_id] = {
                "centroid": (centroid[0], centroid[1]),
                "box": centroid[2],
                "age": 0
            }
            self.next_id += 1

        self.tracks = updated_tracks
        return list(self.tracks.items())

tracker = CentroidTracker(max_age=30, max_distance=30)

def crop_plate(frame, box):
    """Crop license plate from frame."""
    x_min, y_min, x_max, y_max = box
    return frame[int(y_min):int(y_max), int(x_min):int(x_max)]

def save_plate(plate_img, track_id):
    """Save cropped plate image."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"plate_{timestamp}_{track_id}.png"
    filepath = os.path.join(CROPPED_PLATES_DIR, filename)
    cv2.imwrite(filepath, plate_img)
    print(f"Saved plate to {filepath}")
    return filepath

def recognize_plate_with_easyocr(plate_img):
    """Recognize plate number using EasyOCR with validation."""
    try:
        # Enhance the image before OCR to improve accuracy
        plate_img = enhance_image(plate_img)
        result = reader.readtext(plate_img, detail=0)
        if result:
            plate_number = " ".join(result).strip()
            # Extract only digits
            cleaned_plate = ''.join(char for char in plate_number if char.isdigit())
            # Validate digit count: 6 for civil, 8 for government
            if len(cleaned_plate) not in [6,7, 8,9] or not cleaned_plate.isdigit():
                return "Invalid plate format"
            return cleaned_plate  # Return only the digits
        return "No text detected"
    except Exception as e:
        print(f"Error in EasyOCR recognition: {e}")
        return "OCR error"

def format_plate_text(digits):
    """Format plate number for Tunisian plates and classify."""
    # Classify based on digit count
    if len(digits) == 7 or len(digits) == 6 or len(digits) == 9:
        # Civil plate: NN TUN NNNN
        first_part = digits[4:]
        last_part = digits[:4]
        return f"{first_part} تونس {last_part} (Civil)"
    elif len(digits) == 8:
        # Government plate: NN-NNNNNN
        prefix = digits[:2]
        last_part = digits[2:]
        ministry = GOVERNMENT_PLATES.get(prefix, "Unknown Ministry")
        return f"{prefix}-{last_part} (Government: {ministry})"
    return "Invalid digit count"

def log_plate(plate_number, track_id):
    """Log recognized plate to file and console with validation."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Skip logging if plate_number is invalid
    if plate_number in ["No text detected", "OCR error", "Invalid plate format"]:
        log_entry = f"{timestamp} | Track ID: {track_id} | Plate: {plate_number} (Skipped due to invalid format)\n"
        print(log_entry.strip())
        with open(OUTPUT_LOG, "a", encoding='utf-8') as f:
            f.write(log_entry)
        return

    # Validate digit count
    digits = ''.join(filter(str.isdigit, plate_number))
    if len(digits) not in [7, 8]:
        log_entry = f"{timestamp} | Track ID: {track_id} | Plate: {plate_number} (Skipped: Expected 6 or 8 digits, got {len(digits)})\n"
        print(log_entry.strip())
        with open(OUTPUT_LOG, "a", encoding='utf-8') as f:
            f.write(log_entry)
        return

    # Format and log the plate
    formatted_plate = format_plate_text(digits)
    log_entry = f"{timestamp} | Track ID: {track_id} | Plate: {formatted_plate}\n"
    print(log_entry.strip())
    with open(OUTPUT_LOG, "a", encoding='utf-8') as f:
        f.write(log_entry)

def main():
    # Read the input image
    frame = cv2.imread(INPUT_SOURCE)
    if frame is None:
        print(f"Error: Could not open {INPUT_SOURCE}. Check file path.")
        return

    print("Processing image")
    # Run YOLOv8 inference
    results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    print(f"Detected {len(results[0].boxes)} boxes.")
    boxes = []

    # Prepare detections
    for box in results[0].boxes:
        x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
        boxes.append([x_min, y_min, x_max, y_max])

    # Update tracker
    tracks = tracker.update(boxes)
    print(f"Tracker returned {len(tracks)} tracks.")

    # Process each tracked plate
    for track_id, track_data in tracks:
        ltrb = track_data["box"]
        x_min, y_min, x_max, y_max = map(int, ltrb)

        # Crop plate
        plate_img = crop_plate(frame, ltrb)

        # Save cropped plate
        plate_filepath = save_plate(plate_img, track_id)

        # Recognize plate number
        plate_number = recognize_plate_with_easyocr(plate_img)
        if plate_number and plate_number != "No text detected" and plate_number != "OCR error":
            log_plate(plate_number, track_id)

            # Format the plate for display
            formatted_plate = format_plate_text(plate_number)
            # Overlay plate number on frame
            cv2.putText(frame, f"ID: {track_id} Plate: {formatted_plate}",
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Display image
    cv2.imshow("License Plate Detection", frame)
    cv2.waitKey(0)  # Wait for any key press to close
    cv2.destroyAllWindows()
    print("Script completed.")

if __name__ == "__main__":
    main()