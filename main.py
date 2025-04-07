from browser_use import Agent, Browser
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio
from dotenv import load_dotenv
import os
import logging
import time
from pathlib import Path
from pydantic import SecretStr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("browser_automation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("deepmentor_login")

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables or .env file")

# Set up display for headless environments
os.environ['DISPLAY'] = ':99'

# Login information
LOGIN_URL = "http://192.168.0.200:5173"
# LOGIN_TASK = "Log in to the DeepMentor platform using login_email and login_password. After logging in, wait for the dashboard to load. and click the setting btn add new account with your thought,notice you need to remember the new account need to be the guest permissions. and collection permission need to check en ,and check account list is success or not if not you need to try until success!"
# UPLOAD_FILE_TASk = 'You are a helpful agent need to follow the steps below, Log in to the DeepMentor platform using login_email and login_password,if error please enter it again until it correct. After logging in, wait for the dashboard to load.and click the setting btn,and goto files & collections upload files,go to the download page, and select the files which is download_data.csv,then upload it,until upload success!'
RAG_TASK = 'You are a helpful agent need to follow the steps below, Log in to the DeepMentor platform using {login_email} and {login_password = password},if error please delete all info and enter it again until it correct.Goto the document expert page,choose Conversation Mode to RAG mode and click choose collection and toggle en select,then enter hello in the textarea block<placeholder:Ask anything about this collection>, and click the icon btn to start chat'

gpt= ChatOpenAI(model='gpt-4o', api_key=openai_api_key,temperature=0.0)
gemini = ChatGoogleGenerativeAI(model='gemini-2.5-pro-preview-03-25', api_key=SecretStr(os.getenv('GEMINI_API_KEY')))



async def run_login_automation():
    # Ensure logs directory exists
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure browser settings
    config = BrowserContextConfig(
        cookies_file=None,
        wait_for_network_idle_page_load_time=5.0,  # Increased for better page loading
        browser_window_size={'width': 1200, 'height': 800},
        locale='en-US',
        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36',
        highlight_elements=True,
        viewport_expansion=500,
    )
    
    # Initialize browser instances
    browser = None
    context = None
    
    try:
        logger.info("Initializing browser...")
        browser = Browser()
        context = BrowserContext(browser=browser, config=config)
        
        # Define initial navigation action
        initial_actions = [
            {'open_tab': {'url': LOGIN_URL}},
        ]
        
        # Define sensitive data with placeholders
        sensitive_data = {
            'login_email': 'admin@deepmentor.ai',
            'login_password': 'password',
            'new_signup_name': 'BrowserUSE',
            'new_signup_email': 'admin10@deepmentor.ai',
            'new_signup_password': 'Aapassword123',
        }
        
        # Create timestamp for log filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_filename = f"login_session_{timestamp}.json"
        
        # Initialize the agent with the browser context
        logger.info("Creating automation agent...")
        agent = Agent(
            initial_actions=initial_actions,
            sensitive_data=sensitive_data,
            browser_context=context,
            task=RAG_TASK,
            llm=gpt,
            use_vision=True,
            save_conversation_path=str(log_dir / log_filename)
        )
        
        # Run the agent
        logger.info("Starting login automation...")
        history = await agent.run()
        
        # Get final result
        result = history.final_result()
        logger.info(f"Automation completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error during automation: {e}", exc_info=True)
        return f"Failed: {str(e)}"
    
    finally:
        # Ensure proper cleanup
        if context:
            logger.info("Closing browser context...")
            await context.close()
        if browser:
            logger.info("Closing browser...")
            await browser.close()

if __name__ == '__main__':
    logger.info("Starting DeepMentor login automation...")
    try:
        result = asyncio.run(run_login_automation())
        logger.info(f"Final result: {result}")
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
    logger.info("Automation process completed")
