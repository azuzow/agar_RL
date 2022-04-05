import cv2
import pytesseract
import re
import numpy as np

def img2score(img, username,timestep):
    h,w,c = img.shape

    if username == "steph":
        game_height = 1280
        game_width = 2400

        lb_x1 = 2025
        lb_x2 = 2500
        lb_y1 = 15
        lb_y2 = 460

        score_x1 = 15
        score_x2 = 156
        score_y1 = 1222
        score_y2 = 1265
    elif username == "alex":
        game_height = 1154
        game_width = 1221

        lb_x1 = 1020
        lb_x2 = 1211
        lb_y1 = 9
        lb_y2 = 245

        score_x1 = 17
        score_x2 = 100
        score_y1 = 1120
        score_y2 = 1138
    else:
        assert False

    if not (h == game_height and w == game_width):
        return None, None, True

    #mask out leader boards
    cv2.rectangle(img,(lb_x1,lb_y1),(lb_x2,lb_y2),(255,255,255),-1)

    score = img[score_y1:score_y2, score_x1:score_x2]

    # score = cv2.cvtColor(score, cv2.COLOR_BGR2GRAY)
    lower = np.array([0,0,0], dtype = "uint16")
    upper = np.array([235,235,235], dtype = "uint16")
    score = cv2.inRange(score,lower,upper)
    
    
    # score = cv2.adaptiveThreshold(score,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    # score =  cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # score = 255-score
    # score = cv2.GaussianBlur(score,(5,5),0)

    # score = cv2.cvtColor(score, cv2.COLOR_BGR2GRAY)

    # ret, score =cv2.threshold(score,30,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
   
    # score = cv2.cvtColor(score, cv2.COLOR_BGR2GRAY)
    # score= cv2.dilate(score, kernel, iterations = 1)
    # score= cv2.morphologyEx(score, cv2.MORPH_OPEN, kernel)
    # score = cv2.erode(score, kernel, iterations = 1)
    # score =cv2.Canny(score, 50, 100)
    # score = cv2.dilate(score, kernel)
    # score=cv2.erode(score,kernel)
   
    path = '/home/alexzuzow/Desktop/agar_multiagent/scores/score'+str(timestep)+'.png'
    cv2.imwrite(path,score)
    score_str = pytesseract.image_to_string(score)
    # print (score_str)
    # cv2.imshow("score",score)
    # cv2.waitKey(0)


    if "score" in score_str.lower():
        try:
            score = int(re.findall(r'\d+',score_str)[0])
        except IndexError:
            return None, None, True
        failed = False
    else:
        #TODO: could trigger early if blob is in score label region
        score = None
        failed = True
    #mask out score
    cv2.rectangle(img,(score_x1,score_y1),(score_x2,score_y2),(255,255,255),-1)


    return img,score,failed
#
# img = cv2.imread("/Users/stephanehatgiskessell/Downloads/agent_observations/94.png")
# img, score, done = img2score(img,"alex")
# print (score,done)
