import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

APP_URL = os.environ["APP_URL"]

options = Options()
options.add_argument("--headless=new")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(options=options)

try:
    print(f"Loading {APP_URL}")
    driver.get(APP_URL)

    # Give the app time to render / detect sleep state
    time.sleep(5)

    # Look for the "wake up" button Streamlit Cloud shows when app is asleep
    try:
        wake_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'get this app back up')]"))
        )
        print("App is asleep — clicking wake button")
        wake_button.click()
        time.sleep(15)  # allow reboot time
    except Exception:
        print("No wake button found — app likely already awake")

    print("Page title:", driver.title)

finally:
    driver.quit()
