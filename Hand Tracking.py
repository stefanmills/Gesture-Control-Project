import cv2,time
import mediapipe as mp

#creating video object
camera=cv2.VideoCapture(0)
add="https://192.168.43.1:8080/video"
camera.open(add)

#create module from mediapipe
#mp.solution section is a formality on has to do
mpHands=mp.solutions.hands

#hand class  has default parameter of False static image,
# 2 hands detection,mindetection=0.5, min_tracking_confidence=0.5
hands=mpHands.Hands()

#drawing lines on the hand
mpDraw=mp.solutions.drawing_utils

pTime=0
cTime=0

while True:
    ret,img=camera.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #hands variable uses an RGB inmage only
    result=hands.process(gray)
    #print(result.multi_hand_landmarks)

    #checking if we have multiple hands
    if result.multi_hand_landmarks:
        for handLand in result.multi_hand_landmarks:

            #Drawing hands
            mpDraw.draw_landmarks(img,handLand,mpHands.HAND_CONNECTIONS)


    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img, str(int(fps)),(20,70),cv2.FONT_HERSHEY_PLAIN,2,
    (255,255,255),3)

    cv2.imshow("Hand Tracking",img)
    if cv2.waitKey(1)==27:
        break
