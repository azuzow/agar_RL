import cv2
import pytesseract
import re
import numpy as np


def format_term_img(img):
    #mask out bottom bar
    cv2.rectangle(img,(0,1128),(1221,1154),(255,255,255),-1)
    #mask out leaderboard
    cv2.rectangle(img,(1020,7),(1212,250),(255,255,255),-1)
    return img


def format_frame (img, username,get_score=False):
    h,w,c = img.shape

    if username == "steph":
        game_height = 1286
        game_width = 2400

        lb_x1 = 2025
        lb_x2 = 2500
        lb_y1 = 15
        lb_y2 = 460

        score_x1 = 18
        score_x2 = 153
        score_y1 = 1231
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
        print (h,w)
        return None, None, True

    #mask out leader boards
    cv2.rectangle(img,(lb_x1,lb_y1),(lb_x2,lb_y2),(255,255,255),-1)


    if get_score:
        score = img[score_y1:score_y2, score_x1:score_x2]

        # score = cv2.cvtColor(score, cv2.COLOR_BGR2GRAY)
        lower = np.array([0,0,0], dtype = "uint16")
        upper = np.array([235,235,235], dtype = "uint16")
        score = cv2.inRange(score,lower,upper)
        score_str = pytesseract.image_to_string(score)

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

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if get_score:
        return img,score,failed
    else:
        return img

def img2score(img, username,timestep):

    return format_frame (img, username,get_score=True)
# #
# img = cv2.imread("agent_observations/4.png")
# # #
# # # cv2.imshow("img",img)
# # # cv2.waitKey(0)
# img, score, done = img2score(img,"steph",1)
# print (score,done)
