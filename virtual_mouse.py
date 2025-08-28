import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize Mediapipe Hand model
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# Screen size
screen_w, screen_h = pyautogui.size()

# Open webcam
cap = cv2.VideoCapture(0)

# For smoothing cursor movement
prev_x, prev_y = 0, 0
smoothening = 5

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Mirror view
    h, w, c = frame.shape
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            
            # Index fingertip (landmark 8) coordinates
            x = int(lm[8].x * w)
            y = int(lm[8].y * h)
            
            # Map to screen size
            screen_x = np.interp(x, (0, w), (0, screen_w))
            screen_y = np.interp(y, (0, h), (0, screen_h))
            
            # Smooth cursor movement
            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y
            
            # Draw fingertip
            cv2.circle(frame, (x, y), 10, (255, 0, 255), -1)
            
            # Thumb tip (landmark 4)
            thumb_x = int(lm[4].x * w)
            thumb_y = int(lm[4].y * h)
            
            cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 255, 0), -1)
            
            # Distance between thumb & index → click action
            distance = np.hypot(thumb_x - x, thumb_y - y)
            if distance < 40:  # Adjust threshold
                pyautogui.click()
                cv2.putText(frame, "Click", (x, y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw full hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow("Virtual Mouse", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
