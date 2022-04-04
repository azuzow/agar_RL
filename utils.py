import cv2
import pytesseract
import re


def img2score(img, username):
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

        score_x1 = 15
        score_x2 = 90
        score_y1 = 1117
        score_y2 = 1140
    else:
        assert False

    if not (h == game_height and w == game_width):
        return None, None, None

    #mask out leader boards
    cv2.rectangle(img,(lb_x1,lb_y1),(lb_x2,lb_y2),(255,255,255),-1)

    score = img[score_y1:score_y2, score_x1:score_x2]
    score_str = pytesseract.image_to_string(score)
    if "score" in score_str.lower():
        score = int(re.findall(r'\d+',score_str.split(" ")[1].strip())[0])
        done = False
    else:
        #TODO: could trigger early if blob is in score label region
        score = None
        done = True
    #mask out score
    cv2.rectangle(img,(score_x1,score_y1),(score_x2,score_y2),(255,255,255),-1)


    return img,score,done

img = cv2.imread("agent_observations/6.png")
img, score, done = img2score(img,"alex")
print (score,done)
cv2.imshow("img",img)
cv2.waitKey(0)
