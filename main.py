import cv2
import threading
from services import FaceRecognitionService, VoiceToText
import numpy as np
import speech_recognition as sr
from db import engine, EmbeddingModel
from sqlalchemy.orm import sessionmaker

session = sessionmaker(bind=engine)
Session = session()
# Initialize face recognition and voice-to-text services
face_recognition_service = FaceRecognitionService()
# voice_to_text_service = VoiceToText("turbo")

# Global variable to store recognized speech
text_input = ""
cap = cv2.VideoCapture("./teste.mp4")
cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)


def calculate_cosine_similarity(emb1: np.array, emb2: np.array):
    cos_sim = np.dot(emb1, emb2.T) / \
        (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return cos_sim


def embedding_to_text(emb: np.array) -> str:
    # Convert the numpy array to a string representation
    text_emb = ','.join([str(value) for value in emb])
    return text_emb


def text_to_embedding(emb_text: str) -> np.array:
    # Convert the string back to a numpy array
    values = [float(value) for value in emb_text.split(',')]
    return np.array(values)


def listen_to_microphone():
    global text_input
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Adjusting for ambient noise... please wait")
        recognizer.adjust_for_ambient_noise(source)
        print("Microphone ready!")

    while True:
        with mic as source:
            print("Listening...")
            audio = recognizer.listen(source, timeout=30)

        try:
            # Recognize speech using Google Speech Recognition
            text = recognizer.recognize_google(audio, language="pt-BR")
            print("You said: ", text)
            text_input = text  # Update global text_input
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(
                f"Could not request results from Google Speech Recognition service; {e}")


def draw_bboxes(bboxes: np.array, frame: np.array, names: dict) -> np.array:
    """
    Draw bounding boxes and corresponding names on the frame.
    Arguments:
    - bboxes: Array of bounding boxes.
    - frame: The current video frame.
    - names: Dictionary of names mapped to bounding box indices.
    """
    for i, box in enumerate(bboxes):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw name above the bounding box if it exists
        if i in names:
            cv2.putText(frame, names[i], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


# Start microphone listening thread
microphone_thread = threading.Thread(target=listen_to_microphone, daemon=True)
microphone_thread.start()

# Retrieve embeddings from the database
db_embeddings = Session.query(EmbeddingModel).all()
treated_retrieved_embeddings = {}
for emb in db_embeddings:
    treated_retrieved_embeddings[emb.name] = text_to_embedding(emb.embedding)


def get_medium_bbox(bboxes: np.array) -> np.array:
    if len(bboxes) == 0:
        return None
    # Sort bboxes by x-coordinate (left to right)
    sorted_bboxes = sorted(bboxes, key=lambda x: x[0])
    return sorted_bboxes[len(sorted_bboxes) // 2]


# Plausible positions based on the x-coordinate
plausable_positions = {"esquerda": lambda x: np.argmin(x[:, 0]),
                       "meio": get_medium_bbox,
                       "direita": lambda x: np.argmax(x[:, 0])}

recognized_names = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Generate embeddings and bounding boxes
    embeddings, error, bboxes = face_recognition_service.generate_embedding(
        frame)
    if error:
        print(f"Error: {error}")
        continue

    # Handle the speech input (e.g., "Esse é o Paulo, na esquerda")
    if "Esse é o" in text_input or "Essa é a" in text_input:
        name = text_input.split()[-3]
        if len(bboxes) > 1:
            position = ""
            for plausable_position in plausable_positions:
                if plausable_position in text_input:
                    position = plausable_position
            if position:
                # Get the selected bounding box based on the spoken position
                selected_bbox = plausable_positions[position](bboxes)
                if selected_bbox is not None:
                    new_embedding = EmbeddingModel(
                        name=name, embedding=embedding_to_text(embeddings[selected_bbox]))
                    Session.add(new_embedding)
                    Session.commit()
                    recognized_names[selected_bbox] = name
        else:
            new_embedding = EmbeddingModel(
                name=name, embedding=embedding_to_text(embeddings[0]))
            print(f"Created new embedding for: {name}")
            Session.add(new_embedding)
            Session.commit()
            recognized_names[0] = name

    # Compare new embeddings to existing ones
    for i, emb in enumerate(embeddings):
        for db_name, db_emb in treated_retrieved_embeddings.items():
            similarity = calculate_cosine_similarity(emb, db_emb)
            if similarity > 0.95:  # Assume threshold for a match
                recognized_names[i] = db_name

    # Draw bounding boxes and names on the frame
    draw_bboxes(bboxes, frame, recognized_names)

    # Display the video feed
    cv2.imshow("Webcam", frame)

    # Check for 'q' key to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    text_input = ""

# Clean up resources
cap.release()
cv2.destroyAllWindows()
