
import cv2,time
import mediapipe as mp

class HandDetector():
    def __init__(self,mode=False, maxHands=2,detectionCon=0.5,trackCon=0.5 ):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionCon=detectionCon
        self.trackCon=trackCon

        #create module from mediapipe
        #mp.solution section is a formality on has to do
        self.mpHands=mp.solutions.hands

#hand class  has default parameter of False static image,
# 2 hands detection,mindetection=0.5, min_tracking_confidence=0.5
        self.hands=self.mpHands.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)

#drawing lines on the hand
        self.mpDraw=mp.solutions.drawing_utils
    def findHands(self,img,draw=True):
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #hands variable uses an RGB inmage only
        self.result=self.hands.process(gray)
    #print(result.multi_hand_landmarks)

    #checking if we have multiple hands
        if self.result.multi_hand_landmarks:
            for handLand in self.result.multi_hand_landmarks:
                if draw:
            #Drawing hands
                    self.mpDraw.draw_landmarks(img,handLand,self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self,img,handNum=0,draw=True):

        landlist=[]

        if self.result.multi_hand_landmarks:

            myhand=self.result.multi_hand_landmarks[handNum]

            for id,lm in enumerate(myhand.landmark):

                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y* h)
                print(id,cx,cy)
                landlist.append([id,cx,cy])

                if draw:
                    cv2.circle(img,(cx,cy),3,(0,0,255), cv2.FILLED)

        return landlist


def main():

    pTime=0
    cTime=0

    camera=cv2.VideoCapture(0)
    add="https://192.168.43.1:8080/video"
    camera.open(add)
    detector=HandDetector()
    while True:
        ret,img=camera.read()
        img=detector.findHands(img)
        landlist=detector.findPosition(img)

        if len(landlist)!=0:
            print(landlist[4])
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime

        cv2.putText(img, str(int(fps)),(20,70),cv2.FONT_HERSHEY_PLAIN,2,
        (255,255,255),3)

        cv2.imshow("Hand Tracking",img)
        if cv2.waitKey(1)==27:
            break


if __name__=="__main__":
    main()