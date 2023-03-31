import cv2
import mediapipe as mp
import math

class Button:
    def __init__(self, pos, width, height, value):
        # initialize Button object with position, width, height and value attributes
        self.pos = pos
        self.width = width
        self.height = height
        self.value = value
        # set default color and hover_color values
        self.color = (203, 192, 255)
        self.hover_color = (255, 255, 255) 
        # set initial clicked value to False
        self.clicked = False 

    def draw(self, img):
        # if button is clicked, draw the button with hover_color
        if self.clicked:
            cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height), self.hover_color, cv2.FILLED)
        # if button is not clicked, draw the button with default color
        else:
            cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height), self.color, cv2.FILLED)
        # draw the border of the button with a black color and thickness 3
        cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height),(200, 50, 200), 3)
        # add the value text on the button with font size 2, and thickness 2 in black color
        cv2.putText(img, self.value, (self.pos[0] + 40, self.pos[1] + 60), cv2.FONT_HERSHEY_PLAIN,2, (200, 50, 200), 2)

    
    def click(self, pos):
        # Check if the given position (pos) is within the button boundaries
        if self.pos[0] < pos[0] < self.pos[0] + self.width and self.pos[1] < pos[1] < self.pos[1] + self.height:
            # If the position is within the button boundaries, mark the button as clicked and return its value
            self.clicked = True
            return self.value
        else:
            # If the position is not within the button boundaries, mark the button as not clicked and return an empty string
            self.clicked = False
            return ""

    def distance(self, point1, point2):
        # Calculate the Euclidean distance between two 3D points (point1 and point2)
        x1, y1, z1 = point1.x, point1.y, point1.z
        x2, y2, z2 = point2.x, point2.y, point2.z
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)


    
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
buttonListValues = [['W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
                    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
                    ['Z', 'X', 'C', 'V', 'B', 'N', 'M', 'Q', '-'],]
buttonList = []
equation = ""

for x in range(10):
    for y in range(3):
        xpos = x * 100 + 100
        ypos = y * 100 + 150
        buttonList.append(Button((xpos, ypos), 100, 100, buttonListValues[y][x]))

with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    last_button_value = ""
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # get the landmark locations for the hand
                finger_tip_locations = []
                for id in range(4, 6):
                    finger_tip_locations.append(hand_landmarks.landmark[id].x * img.shape[1])
                    finger_tip_locations.append(hand_landmarks.landmark[id].y * img.shape[0])
                # check if any button is clicked
                button_value = ""
                for button in buttonList:
                    if button.click(finger_tip_locations) and button.distance(hand_landmarks.landmark[8], hand_landmarks.landmark[12]) < 0.05:
                        button_value = button.value
                        break
                # draw circle on finger tips
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
                    if landmark == hand_landmarks.landmark[8] or landmark == hand_landmarks.landmark[12]:
                        cv2.circle(img, (x, y), 10, (100, 100, 0), cv2.FILLED)
                # update the equation
                if button_value != "":
                    if button_value == "-":
                        equation = equation[:-1] # remove the last character from the equation
                        last_button_value = button_value
                    else:
                        if last_button_value != button_value:
                            equation += button_value
                            last_button_value = button_value
                break
        cv2.rectangle(img, (100, 70), (800 + 200, 70 + 100), (225, 225, 225), cv2.FILLED)
        cv2.rectangle(img, (100, 70), (800 + 200, 70 + 100), (200, 50, 200), 3)
        for button in buttonList:
            button.draw(img)
        cv2.putText(img, equation, (110, 130), cv2.FONT_HERSHEY_PLAIN, 3, (200, 50, 200), 3)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
