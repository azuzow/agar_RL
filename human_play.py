from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
import time


from agar import env
import utils
from utils import ReplayMemory, Net

path_to_adblock = r'1.42.2_0'
chrome_options = webdriver.ChromeOptions()
chrome_options.add_experimental_option('useAutomationExtension', False)
chrome_options.add_argument('--disable-blink-features=AutomationControlled')
chrome_options.add_argument('load-extension=' + path_to_adblock)
name='cs394r'

agar1 = env(chrome_options,name)
episode = 0
while True:
    state = agar1.reset()
    time.sleep(.5)
    timestep = 0
    while True:
        action = 0
        next_state,score,failed,restart,done = agar1.step(action,timestep,episode)

        timestep+=1
        if  restart:
            episode += 1
            break
