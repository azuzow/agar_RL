import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image
import time
import timeit
import cv2
import os
import io
import torch
import torchvision

from utils import img2score,format_frame,format_term_img

class env:
    def __init__(self,chrome_options,name):
        self.driver = None
        self.chrome_options=chrome_options
        self.action_selector = None
        self.screen_width=None
        self.screen_height=None
        actions=np.linspace(1,360,50)
        actions = np.radians(actions)
        x_actions= 200*np.cos(actions)
        y_actions= 200*np.sin(actions)

        self.actions_taken=[]
        self.name=name
        self.action_space = dict()
        self.action_space[0]= Keys.SPACE
        for i in range(len(actions)):
            self.action_space[i+1]= tuple((x_actions[i],y_actions[i]))
        self.n_fails = 0
        self.first_fail_frames = []

    def reset(self):
        frames=None
        try:
            self.driver.quit()
        except:
            pass
        self.actions_taken=[]
        self.steps_no_score = 0
        self.n_fails = 0
        try:
            self.driver = webdriver.Chrome(ChromeDriverManager().install(),options=self.chrome_options)
            self.action_selector = ActionChains(self.driver,duration=0)
            # self.action_selector = ActionChains(self.driver)
            self.driver.set_window_size(1000, 1000)
            # self.action_selector.duration = 0


            self.driver.get("https://agar.io/#ffa")
            menu = WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable((By.XPATH,'//*[@id="nick"]')))
            menu.send_keys(self.name)
            play_button = WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable((By.XPATH,'//*[@id="play"]')))
            play_button.click()
            game_screen = self.driver.find_element(By.XPATH,'//*[@id="canvas"]')
            self.screen_height=int(game_screen.get_attribute('height'))
            self.screen_width=int(game_screen.get_attribute('width'))

            # self.action_selector.move_by_offset(self.screen_width/2,self.screen_height/2).perform()
            self.action_selector.move_to_element(game_screen)


            obs= self.get_screenshot("f1.png")
            masked_img_1 = format_frame (obs, "alex",self.n_fails)
            masked_img_1 = torchvision.transforms.functional.to_tensor(masked_img_1)

            obs= self.get_screenshot("f2.png")
            masked_img_2 = format_frame (obs, "alex",self.n_fails)
            masked_img_2 = torchvision.transforms.functional.to_tensor(masked_img_2)

            obs= self.get_screenshot("f3.png")
            masked_img_3 = format_frame (obs, "alex",self.n_fails)
            masked_img_3 = torchvision.transforms.functional.to_tensor(masked_img_3)

            frames = torch.cat((masked_img_1, masked_img_2, masked_img_3))
            return frames
        except Exception as e:
            print(e)
            self.reset()

        return frames

    # def get_screenshot(self,obs_path):
    #     # obs = io.BytesIO(self.driver.get_screenshot_as_png())
    #     # obs =Image.open(obs)
    #     self.driver.save_screenshot(obs_path)
    #     img = cv2.imread(obs_path)
    #     obs = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     return obs

    def get_screenshot(self,obs_path):
        # obs = io.BytesIO(self.driver.get_screenshot_as_png())
        # obs =Image.open(obs)
        img = self.driver.get_screenshot_as_png()
        img = np.frombuffer(img, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        obs = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return obs

    def step(self,action,timestep,episode):

        obs_path= 'agent_observations/'+ str(timestep+episode)+'.png'
        print(action)

        if action==0:
            print(self.action_space[action])
            self.action_selector.send_keys(self.action_space[action]).perform()
        else:

            self.actions_taken.append(self.action_space[action])
            #move cursor back to center of screen
            # tic = timeit.default_timer()
            if len(self.actions_taken)>1:
                x_offset = self.actions_taken[-1][0]-self.actions_taken[-2][0]
                y_offset= self.actions_taken[-1][1]-self.actions_taken[-2][1]
                self.action_selector.move_by_offset(x_offset,y_offset).perform()
            else:
                print(self.actions_taken[0][0],self.actions_taken[0][1])
                self.action_selector.move_by_offset(self.actions_taken[0][0],self.actions_taken[0][1]).perform()


        frames = []

        obs = self.get_screenshot(obs_path)
        masked_img,score,failed = img2score(obs,"alex",timestep,self.n_fails)
        if not failed:
            masked_img = torchvision.transforms.functional.to_tensor(masked_img)
        else:
            masked_img = torch.zeros((1,128,128))


        obs_1 = self.get_screenshot("f1.png")
        masked_img_1 = format_frame (obs_1, "alex",self.n_fails)
        masked_img_1 = torchvision.transforms.functional.to_tensor(masked_img_1)


        obs_2 = self.get_screenshot("f2.png")
        masked_img_2 = format_frame (obs_2, "alex",self.n_fails)
        masked_img_2 = torchvision.transforms.functional.to_tensor(masked_img_2)
        frames = torch.cat((masked_img,masked_img_1,masked_img_2))

        # obs_3 = self.get_screenshot("f3.png")
        # masked_img_3 = format_frame (obs_3, "steph")
        # frames.append(masked_img_3)

        # masked_img,score,failed = img2score(obs,"alex",timestep)

        print ("Score: ",score)

        restart = False
        done = False

        if failed:
            if self.n_fails == 0 and timestep >= 3:
                self.first_fail_frames = frames
                masked_img = format_term_img(obs)
                masked_img = torchvision.transforms.functional.to_tensor(masked_img)
                self.first_fail_frames[0] = masked_img
                # cv2.imwrite(obs_path,masked_img)

            self.n_fails+=1
            # if self.n_fails >1:
            #     os.remove(obs_path)
            if self.n_fails >=3:
                if timestep >= 6:
                    done = True

                frames = self.first_fail_frames
                restart=True
        else:
            self.n_fails=0
            # cv2.imwrite(obs_path,masked_img)

        return frames,score,failed,restart,done
