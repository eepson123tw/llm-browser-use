from browser_use import Agent, Browser
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from browser_use.browser.browser import BrowserConfig
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import VLLMOpenAI

import asyncio
from dotenv import load_dotenv
import os
import logging
import time
from pydantic import SecretStr
from pathlib import Path
import asyncio
import json
import base64

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

READ_PDF_TASK = '''
you are a good agent, if you finish the steps below you will can $1000000 usd.

search Google Drive and
Search for the file named GitHub Repository Summary within Google Drive.
Click on the GitHub Repository Summary file or use https://docs.google.com/spreadsheets/d/1vcQjfjfW_LULAn8i7FIllbRreaHqb6pRNO7DPBxdg04/edit?usp=drive_link to open it.
Inside the opened CSV file, find the tab corresponding to the date 2025-03-27.
Determine the summary content:
If you found the tab for 2025-03-27, identify the parse the button with the text '生成這個表格的摘要' or '分析這項資料' and copy the summary content.
If you did not find the date 2025-03-27, create a summary based on the overall content visible in the CSV file.



Open your gmail page or web mail client.Start composing a new email message.
no need to enter the to field,let user enter it
1. In the 'Subject' field, enter the text: Hi, I am browser use.
2. In the body of the email, paste the summary text that you obtained in step 5.
it is very important you need to paste the summary text that you obtained in step 5.

Prepare the email and no click send button just need to save to drafts

'''

gpt= ChatOpenAI(model='gpt-4o', api_key=openai_api_key,temperature=0.0)
gemini = ChatGoogleGenerativeAI(model='gemini-2.0-flash', api_key=SecretStr(os.getenv('GEMINI_API_KEY')))
remote_llm = ChatOpenAI(
    api_key="VLLM",
    base_url="http://192.168.0.44:8000/v1",
    model="/app/model/watt-tool-70B-GPTQ-INT4", 
    temperature=0,
    model_kwargs={
        "tool_choice": "auto"  # Allows the model to decide when to use tools
    }
)



async def run_login_automation():
    # Ensure logs directory exists
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure browser settings
    config = BrowserContextConfig(
        cookies_file=None,
        wait_for_network_idle_page_load_time=5.0,
        browser_window_size={'width': 1440, 'height': 1080},
        locale='en-US',
        # user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36',
        highlight_elements=True,
        viewport_expansion=500,
    )
    
    # Initialize browser instances
    browser = None
    context = None
    
    try:
        logger.info("Initializing browser...")
        browser = Browser(BrowserConfig(chrome_instance_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless=False,))
        context = BrowserContext(browser=browser, config=config)
        
        # Create timestamp for log filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_filename = f"login_session_{timestamp}.json"
        conversation_log_path = str(log_dir / log_filename)
        
        # Initialize the agent with the browser context
        logger.info("Creating automation agent...")
        agent = Agent(
            browser_context=context,
            task=READ_PDF_TASK,  # Make sure this constant is defined elsewhere
            llm=remote_llm,
            max_actions_per_step=1,
            max_failures=3,
            use_vision=False,  # Changed to True since you're likely doing visual tasks
            generate_gif=False,
            save_conversation_path=conversation_log_path
        )
        
        # Run the agent
        logger.info("Starting login automation...")
        history = await agent.run()
        
        # Get final result
        result = history.final_result()
        
        # Save screenshots to files instead of logging them directly
        screenshots_dir = log_dir / f"screenshots_{timestamp}"
        screenshots_dir.mkdir(exist_ok=True)
        
        screenshot_paths = []
        screenshots = history.screenshots()
        if screenshots and isinstance(screenshots, list):
            for i, screenshot in enumerate(screenshots):
                if screenshot and isinstance(screenshot, str):
                    try:
                        # Check if it starts with base64 prefix, if not, add it
                        if not screenshot.startswith("data:image"):
                            img_data = base64.b64decode(screenshot)
                        else:
                            # Extract the base64 part if it has a prefix
                            img_data = base64.b64decode(screenshot.split(',')[1])
                            
                        img_path = screenshots_dir / f"screenshot_{i}.png"
                        with open(img_path, "wb") as f:
                            f.write(img_data)
                        screenshot_paths.append(str(img_path))
                    except Exception as e:
                        logger.error(f"Error saving screenshot {i}: {e}")
        
        # Get errors
        errors = history.errors()
        error_summary = []
        if errors:
            for i, error in enumerate(errors):
                if isinstance(error, str) and len(error) > 100:
                    # Truncate long error messages for logging
                    error_summary.append(f"Error {i}: {error[:100]}...")
                else:
                    error_summary.append(f"Error {i}: {error}")
        
        # Save detailed logs to a separate file
        detailed_log = {
            "timestamp": timestamp,
            "result": result,
            "screenshot_paths": screenshot_paths,
            "errors": errors
        }
        
        detailed_log_path = log_dir / f"detailed_log_{timestamp}.json"
        with open(detailed_log_path, "w") as f:
            json.dump(detailed_log, f, indent=2, default=str)
        
        # Log summary information
        logger.info(f"Automation completed: Result={result}")
        logger.info(f"Screenshots saved to {screenshots_dir}")
        if errors:
            logger.info(f"Errors: {error_summary}")
        logger.info(f"Detailed log saved to {detailed_log_path}")
        
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
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("automation.log"),
            logging.StreamHandler()
        ]
    )
    
    logger.info("Starting DeepMentor login automation...")
    try:
        result = asyncio.run(run_login_automation())
        logger.info(f"Final result: {result}")
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
    logger.info("Automation process completed")
