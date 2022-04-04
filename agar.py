import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
import time

path_to_adblock = r'/home/alexzuzow/Desktop/agar_multiagent/1.42.2_0'
chrome_options = webdriver.ChromeOptions()
chrome_options.add_experimental_option('useAutomationExtension', False)
chrome_options.add_argument('--disable-blink-features=AutomationControlled')
chrome_options.add_argument('load-extension=' + path_to_adblock)

driver = webdriver.Chrome(ChromeDriverManager().install(),options=chrome_options)
action_selector = ActionChains(driver)
driver.get("https://agar.io/#ffa")
name='agarmang'
menu = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH,'//*[@id="nick"]')))
menu.send_keys(name)
play_button = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH,'//*[@id="play"]')))
play_button.click()

game_screen = driver.find_element(By.XPATH,'//*[@id="canvas"]')

height=int(game_screen.get_attribute('height'))
width=int(game_screen.get_attribute('width'))
#center cursor
action_selector.move_by_offset(width/2,height/2).perform()
actions=np.linspace(1,360,50)
actions = np.radians(actions)
x_actions= 100*np.cos(actions)
y_actions= 100*np.sin(actions)
actions = np.array([x_actions,y_actions])
actions=actions.T
state=0
actions_taken=[]
cursor_x=0
cursor_y=0
while True:

	driver.save_screenshot('/home/alexzuzow/Desktop/agar_multiagent/agent_observations/'+ str(state)+'.png')

	#move cursor back to center of screen
	if len(actions_taken)>0:
		action_selector.move_by_offset(-x_actions[actions_taken[-1]],-y_actions[actions_taken[-1]]).perform()
		
		print(cursor_x,cursor_y)
	actions_taken.append(np.random.randint(50))
	# print(actions_taken[-1],x_actions[actions_taken[-1]],y_actions[actions_taken[-1]])
	action_selector.move_by_offset(x_actions[actions_taken[-1]],y_actions[actions_taken[-1]]).perform()


	# time.sleep(1)

	state+=1
	if state == 50:
		break

