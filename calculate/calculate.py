import cv2
import mediapipe as mp

class Button:
    def __init__(self, pos, width, height, value):
        self.pos = pos
        self.width = width
        self.height = height
        self.value = value
 
    def draw(self, img):
        cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height),(225, 225, 225), cv2.FILLED)
        cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height),(50, 50, 50), 3)
        cv2.putText(img, self.value, (self.pos[0] + 40, self.pos[1] + 60), cv2.FONT_HERSHEY_PLAIN,2, (50, 50, 50), 2)
    
    def click(self,x,y):
        if self.pos[0]<x<self.pos[0] + self.width:
            cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height),(255, 255, 255), cv2.FILLED)
            cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height),(50, 50, 50), 3)
            cv2.putText(img, self.value, (self.pos[0] + 40, self.pos[1] + 60), cv2.FONT_HERSHEY_PLAIN,2, (0, 0, 0), 2)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
buttonListValues = [['7', '8', '9', '*'],
                    ['4', '5', '6', '-'],
                    ['1', '2', '3', '+'],
                    ['0', '/', '.', '=']]
buttonList = []
equation = ""
for x in range(4):
    for y in range(4):
        xpos = x * 100 + 800
        ypos = y * 100 + 150
        buttonList.append(Button((xpos, ypos), 100, 100, buttonListValues[y][x]))

with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                cursor = hand_landmarks.landmark[8]
                middle_finger = hand_landmarks.landmark[12]
                distance = ((middle_finger.x - cursor.x)**2 + (middle_finger.y - cursor.y)**2 + (middle_finger.z - cursor.z)**2)**0.5
                if distance < 0.04:
                    pass
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.rectangle(img, (800, 70), (800 + 400, 70 + 100), (225, 225, 225), cv2.FILLED)
        cv2.rectangle(img, (800, 70), (800 + 400, 70 + 100), (50, 50, 50), 3)
        for button in buttonList:
            button.draw(img)
        cv2.putText(img, equation, (810, 130), cv2.FONT_HERSHEY_PLAIN, 3, (50, 50, 50), 3)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()