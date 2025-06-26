from ultralytics import YOLO
import cv2

# Load a more accurate model (you can also try yolov8s.pt if you want faster)
model = YOLO("yolov8m.pt")  # Automatically downloads if not available

# Open webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

# Define confidence threshold
CONFIDENCE_THRESHOLD = 0.6

while True:
    success, frame = cap.read()
    if not success:
        break

    # Run detection
    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:
            # Extract box info
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # Show only predictions with high confidence
            if conf > CONFIDENCE_THRESHOLD:
                label = f"{class_name} {conf:.2f}"
                color = (0, 255, 0)  # Green box

                # Draw rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("YOLOv8 Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press q to quit
        break

cap.release()
cv2.destroyAllWindows()
