import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()  # Read a frame from the webcam

    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    cv2.imshow("Webcam", frame)  # Display the frame

    # Press 'q' to exit the webcam display
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the VideoCapture object and close windows
cap.release()
cv2.destroyAllWindows()
