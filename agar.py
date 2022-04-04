import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
import time
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
		self.driver = webdriver.Chrome(ChromeDriverManager().install(),options=chrome_options)
		self.action_selector= ActionChains(self.driver)
		self.driver.get("https://agar.io/#ffa")
		menu = WebDriverWait(self.driver, 20).until(EC.element_to_be_clickable((By.XPATH,'//*[@id="nick"]')))
		menu.send_keys(name)
		play_button = WebDriverWait(self.driver, 20).until(EC.element_to_be_clickable((By.XPATH,'//*[@id="play"]')))
		play_button.click()
		game_screen = self.driver.find_element(By.XPATH,'//*[@id="canvas"]')
		self.screen_height=int(game_screen.get_attribute('height'))
		self.screen_width=int(game_screen.get_attribute('width'))
		self.action_selector.move_by_offset(self.screen_width/2,self.screen_height/2).perform()

	def step(self,action):
		obs_path= '/agent_observations/'+ str(state)+'.png'
		self.driver.save_screenshot(obs_path)
		self.actions_taken.append(action)
		#move cursor back to center of screen
		if len(self.actions_taken)>1:
			x_offset = self.x_actions[self.actions_taken[-1]]-self.x_actions[self.actions_taken[-2]]
			y_offset= self.y_actions[self.actions_taken[-1]]-self.y_actions[self.actions_taken[-2]]
			
			self.action_selector.move_by_offset(x_offset,y_offset).perform()
		else:
			self.action_selector.move_by_offset(self.x_actions[self.actions_taken[0]],self.y_actions[self.actions_taken[0]]).perform()

		# score = self.driver.find_element(By.XPATH,'//*[@id="statsGraph"]')

		#check if game is over
		done=False 
		try:
			print('====================================================================')
			exit_button = self.driver.find_element(By.XPATH,'//*[@id="statsContinue"]')
			exit_button.click()
			done=True
		except Exception as e:
			print('====================================================================')
			pass
		reward = 0
		return obs_path,reward,done


path_to_adblock = r'1.42.2_0'
chrome_options = webdriver.ChromeOptions()
chrome_options.add_experimental_option('useAutomationExtension', False)
chrome_options.add_argument('--disable-blink-features=AutomationControlled')
chrome_options.add_argument('load-extension=' + path_to_adblock)

agar1 = env(chrome_options,name)
agar1.reset()

state=0

while True:
	action = np.random.randint(50)
	obs_path,reward,done = agar1.step(action)
	print(done)
	if done:
		agar1.reset()

	state+=1
	# if state == 50:
	# 	break

