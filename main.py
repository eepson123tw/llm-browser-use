from browser_use import Agent, Browser
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from langchain_openai import ChatOpenAI
import asyncio
from dotenv import load_dotenv
import os



# Load variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set up display environment variable
os.environ['DISPLAY'] = ':99'

async def run_search():
    # Configure the browser context
    config = BrowserContextConfig(
        cookies_file=None,
        wait_for_network_idle_page_load_time=3.0,
        browser_window_size={'width': 1280, 'height': 1100},
        locale='en-US',
        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36',
        highlight_elements=True,
        viewport_expansion=500,
        # Or alternatively, include about:blank: 
        # allowed_domains=['about:blank', 'google.com', 'wikipedia.org', 'bing.com', 'news.google.com'],
    )
    
    # Initialize browser and context
    browser = Browser()
    context = BrowserContext(browser=browser, config=config)
    
    # Create the agent with the browser context
    agent = Agent(
        browser_context=context,
        task="Search for the latest news about artificial intelligence and summarize the top 3 articles",
        llm=ChatOpenAI(model='gpt-4o', api_key=openai_api_key),
    )
    
    try:
        print("Running browser agent with unrestricted domain access...")
        history = await agent.run()
        result = history.final_result()
        print(f"{result}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close the browser context and browser
        print("Closing browser context and browser...")
        await context.close()
        await browser.close()
    
    print("Task completed.")

if __name__ == '__main__':
    print("Starting search task...")
    asyncio.run(run_search())
