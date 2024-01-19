import cv2
import mediapipe as mp
import pickle

model_dict = pickle.load(open('./model.pickle', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('video_classifier_4.mp4', fourcc, 20.0, (1280, 720))


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape
    X = []
    Y = []

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        data_aux = []

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                data_aux.append(x)
                data_aux.append(y)

                X.append(x)
                Y.append(y)

        y_pred = model.predict([data_aux])

        # Update the displayed text
        x1 = int(min(X) * width) - 20
        y1 = int(min(Y) * height) - 20
        x2 = int(max(X) * width) + 20
        y2 = int(max(Y) * height) + 20

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, y_pred[0], (x1 + 40, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2,
                    cv2.LINE_AA)

        out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow('frame', frame)
    cv2.waitKey(25)

cap.release()
out.release()  # Release the VideoWriter object
cv2.destroyAllWindows()
