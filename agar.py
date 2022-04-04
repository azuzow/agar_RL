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

		self.driver = webdriver.Chrome(ChromeDriverManager().install(),options=self.chrome_options)


		self.action_selector= ActionChains(self.driver)
		self.driver.get("https://agar.io/#ffa")
		menu = WebDriverWait(self.driver, 20).until(EC.element_to_be_clickable((By.XPATH,'//*[@id="nick"]')))
		menu.send_keys(name)
		play_button = WebDriverWait(self.driver, 20).until(EC.element_to_be_clickable((By.XPATH,'//*[@id="play"]')))
		play_button.click()
		game_screen = self.driver.find_element(By.XPATH,'//*[@id="canvas"]')
		self.screen_height=int(game_screen.get_attribute('height'))
		self.screen_width=int(game_screen.get_attribute('width'))

		# self.action_selector.move_by_offset(self.screen_width/2,self.screen_height/2).perform()
		self.action_selector.move_to_element(game_screen)


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


		# score = self.driver.find_element(By.XPATH,'//*[@id="statsGraph"]')

		#check if game is over
		done=False
		try:
			# print('====================================================================')
			exit_button = self.driver.find_element(By.XPATH,'//*[@id="statsContinue"]')
			exit_button.click()
			done=True
		except Exception as e:
			# print('====================================================================')
			pass


		# tic = timeit.default_timer()
		masked_img,score,failed = img2score(cv2.imread(obs_path),"alex")
		# toc = timeit.default_timer()
		# print ("get score time: " + str(toc-tic))

		restart = False
		if timestep == 1 and failed:
			restart = True
		score = 0
		# failed = False
		# restart = False
		# score = 0
		return obs_path,score,done,failed,restart


path_to_adblock = r'1.42.2_0'
chrome_options = webdriver.ChromeOptions()
chrome_options.add_experimental_option('useAutomationExtension', False)
chrome_options.add_argument('--disable-blink-features=AutomationControlled')
chrome_options.add_argument('load-extension=' + path_to_adblock)
# chrome_options.add_argument("--window-size=1154,1221")

agar1 = env(chrome_options,name)
agar1.reset()

time.sleep(2)

timestep = 0
episode = 0

while True:
	print ("stepping...")
	action = np.random.randint(50)

	# tic = timeit.default_timer()
	obs_path,score,done,failed,restart = agar1.step(action,timestep,episode)
	# toc = timeit.default_timer()
	# print ("step time: " + str(toc-tic))

	if restart:
		print ("NEEDS RESTARTING")
		timestep = 0
		agar1 = env(chrome_options,name)
		agar1.reset()


	if failed:
		timestep+=1
		continue

	print(done,score)
	if done:
		agar1.reset()
		timestep = 0
		episode +=1

	timestep +=1
