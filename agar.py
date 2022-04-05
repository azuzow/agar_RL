import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
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
        self.x_actions= 200*np.cos(actions)
        self.y_actions= 200*np.sin(actions)
        self.actions = np.array([self.x_actions,self.y_actions])
        self.actions=self.actions.T
        self.actions_taken=[]
        self.name=name


    def reset(self):
        try:
            self.driver.quit()
        except:
            pass
        self.actions_taken=[]
        self.steps_no_score = 0
        try:
            self.driver = webdriver.Chrome(ChromeDriverManager().install(),options=self.chrome_options)
            self.action_selector = ActionChains(self.driver,)
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
            print('=======================================')
            print('=======================================')
            print(e)
            self.reset()
            print('=======================================')
            print('=======================================')


    def step(self,action,timestep,episode):

        
        
        obs_path= 'agent_observations/'+ str(timestep+episode)+'.png'
        self.driver.save_screenshot(obs_path)
        self.actions_taken.append(action)
        #move cursor back to center of screen
        # tic = timeit.default_timer()
        if len(self.actions_taken)>1:
            x_offset = self.x_actions[self.actions_taken[-1]]-self.x_actions[self.actions_taken[-2]]
            y_offset= self.y_actions[self.actions_taken[-1]]-self.y_actions[self.actions_taken[-2]]
            self.action_selector.move_by_offset(x_offset,y_offset).perform()
        else:
            self.action_selector.move_by_offset(self.x_actions[self.actions_taken[0]],self.y_actions[self.actions_taken[0]]).perform()

       
        masked_img,score,failed = img2score(cv2.imread(obs_path),"alex",timestep)

        if  failed:
            os.remove(obs_path)
            obs_path=None
       
        return obs_path,score,failed


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
n_fails=0
restart=False
while True:
    
    action = np.random.randint(50)

    # tic = timeit.default_timer()
    obs_path,score,failed = agar1.step(action,timestep,episode)
    # toc = timeit.default_timer()
    # print ("step time: " + str(toc-tic))
    
    if failed:
        n_fails+=1
        print('failed',n_fails,'/',timestep)
        
    else:
        n_fails=0

    if n_fails >=3:
        restart=True


    if  restart:
        agar1.reset()
        n_fails=0
        restart=False
        timestep = 0
        episode +=1
    timestep +=1
    
