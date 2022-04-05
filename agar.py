import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
import time
import timeit
import cv2
import os

from utils import img2score
name='cs394r'

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

    def reset(self):
        try:
            self.driver.quit()
        except:
            pass
        self.actions_taken=[]
        self.steps_no_score = 0
        self.n_fails = 0
        try:
            self.driver = webdriver.Chrome(ChromeDriverManager().install(),options=self.chrome_options)
            # self.action_selector = ActionChains(self.driver,duration=0)
            self.action_selector = ActionChains(self.driver)

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
        except Exception as e:
            print(e)
            self.reset()



    def step(self,action,timestep,episode):



        obs_path= 'agent_observations/'+ str(timestep+episode)+'.png'
        print(action)
        self.driver.save_screenshot(obs_path)
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


        masked_img,score,failed = img2score(cv2.imread(obs_path),"steph",timestep)
        os.remove(obs_path)
        print ("Score: ",score)
        # os.remove(obs_path)
        restart = False

        if  failed:

            self.n_fails+=1
            if self.n_fails >=3:
                restart=True
                if timestep > 10:
                    cv2.imwrite(obs_path,masked_img)
            masked_img = None
        else:
            cv2.imwrite(obs_path,masked_img)
            # os.remove(obs_path)
            # obs_path=None
            self.n_fails=0
        #




        return masked_img,score,failed, restart

path_to_adblock = r'1.42.2_0'
chrome_options = webdriver.ChromeOptions()
chrome_options.add_experimental_option('useAutomationExtension', False)
chrome_options.add_argument('--disable-blink-features=AutomationControlled')
chrome_options.add_argument('load-extension=' + path_to_adblock)

# chrome_options.add_argument("--window-size=1154,1221")

agar1 = env(chrome_options,name)
agar1.reset()

time.sleep(1)

timestep = 0
episode = 0
while True:

    action = np.random.randint(len(agar1.action_space))

    # tic = timeit.default_timer()
    masked_img,score,failed,restart = agar1.step(action,timestep,episode)
    # toc = timeit.default_timer()
    # print ("step time: " + str(toc-tic))


    timestep +=1

    if  restart:
        agar1.reset()
        timestep = 0
        episode +=1
