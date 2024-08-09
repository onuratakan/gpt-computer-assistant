from bs4 import BeautifulSoup
import requests
import re
from urllib.parse import urljoin
import datetime
import traceback

try:
    from .tooler import tool
    from .top_bar_wrapper import wrapper
except:
    from tooler import tool
    from top_bar_wrapper import wrapper

_standard_tools_ = {}

def register_tool(func):
    if func.__name__ not in _standard_tools_:
        _standard_tools_[func.__name__] = tool(func)
    return func
@register_tool
@wrapper
def read_website(url: str, max_content_length: int = 5000) -> dict:
    """
    Read the content of a website and return the title, meta data, content, and sub-links.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        html = response.text
    except requests.RequestException as e:
        return {"error": f"Failed to retrieve the website content: {e}"}

    soup = BeautifulSoup(html, "html.parser")

    meta_properties = [
        "og:description",
        "og:site_name",
        "og:title",
        "og:type",
        "og:url",
        "description",
        "keywords",
        "author"
    ]
    meta = {}
    for property_name in meta_properties:
        tag = soup.find("meta", property=property_name) or soup.find("meta", attrs={"name": property_name})
        if tag:
            meta[property_name] = tag.get("content", "")

    for ignore_tag in soup(["script", "style"]):
        ignore_tag.decompose()

    title = soup.title.string.strip() if soup.title else ""
    content = soup.body.get_text(separator="\n") if soup.body else ""

    links = []
    for a in soup.find_all("a", href=True):
        link_url = urljoin(url, a["href"])
        links.append({"title": a.text.strip(), "link": link_url})

    content = re.sub(r"[\n\r\t]+", "\n", content)
    content = re.sub(r" +", " ", content)
    content = re.sub(r"[\n ]{3,}", "\n\n", content)
    content = content.strip()

    if len(content) > max_content_length:
        content = content[:max_content_length].rsplit(' ', 1)[0] + '...'

    return {"meta": meta, "title": title, "content": content, "sub_links": links}


@register_tool
@wrapper
def google(query: str, max_number: int = 20) -> list:
    """
    Search the query on Google and return the results.
    """
    try:
        from googlesearch import search as gsearch
        return list(gsearch(query, stop=max_number))
    except:
        return "An exception occurred"    


@register_tool
@wrapper
def duckduckgo(query: str, max_number: int = 20) -> list:
    """
    Search the query on DuckDuckGo and return the results.
    """
    try:
        from duckduckgo_search import DDGS
        return [result["href"] for result in DDGS().text(query, max_results=max_number)]
    except:
        return "An exception occurred"



@register_tool
@wrapper
def copy(text: str):
    """
    Copy the text to the clipboard.
    """
    import pyperclip
    pyperclip.copy(text)
    pyperclip.copy(text)


@register_tool
@wrapper
def open_url(url: str) -> bool:
    """
    Open the URL in the default web browser.

    :param url: The URL to open.
    :type url: str
    :return: True if the URL was opened successfully, False otherwise.
    :rtype: bool
    """
    try:
        webbrowser.open(url)
        return True
    except webbrowser.Error as e:
        print(f"Failed to open URL {url}: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

@register_tool
@wrapper
def sleep(seconds: int):
    """
    Sleep for the given number of seconds.
    """
    import time
    time.sleep(seconds)



@register_tool
@wrapper
def keyboard_write(text: str):
    """
    Write the text using the keyboard.
    """
    import pyautogui
    pyautogui.write(text)

@register_tool
@wrapper
def keyboard_press(key: str):
    """
    Press the key using the keyboard.
    """
    import pyautogui
    pyautogui.press(key)
    pyautogui.press(key)



from langchain_experimental.utilities import PythonREPL

the_py_client = PythonREPL()

@register_tool
@wrapper
def python_repl(code: str) -> str:
    """
    Run and return the given python code in python repl
    """
    return the_py_client.run(code)

@register_tool
@wrapper
def app_open(app_name: str) -> bool:
    """
    Opens the native apps.
    """
    try:
        from AppOpener import open
        open(app_name, throw_error=True)
        return True
    except:
        try:
            from MacAppOpener import open
            open(app_name)
        except:
            return False

@register_tool
@wrapper
def app_close(app_name: str) -> bool:
    """
    Closes the native apps.
    """
    try:
        from AppOpener import close
        close(app_name, throw_error=True)
        return True
    except:
        try:
            from MacAppOpener import open
            close(app_name)
        except:
            return False



@register_tool
@wrapper
def get_current_time() -> str:
    """
    Get the current time in ISO format.
    """
    return datetime.datetime.now().isoformat()



@register_tool
@wrapper
def turn_off_wifi() -> bool:
    """
    Turn off the wifi.
    """
    try:
        from pywifi import ControlPeripheral
        wifi = ControlPeripheral()
        wifi.disable()
        return True
    except:

        return False
    
@register_tool
@wrapper
def turn_on_wifi() -> bool:
    """
    Turn on the wifi.
    """
    try:
        from pywifi import ControlPeripheral
        wifi = ControlPeripheral()
        wifi.enable()
        return True
    except:
        
        return False
    

@register_tool
@wrapper
def connect_wifi(ssid: str, password: str) -> bool:
    """
    Connect to the wifi with the given ssid and password.
    """
    try:
        from pywifi import ControlConnection

        # Arguments passed during object instantiation
        controller = ControlConnection(wifi_ssid=ssid, wifi_password=password)
        controller.wifi_connector()
        return True
    except:
        return False




@register_tool
@wrapper
def ask_to_user(question:str, wait_for_answer:str=None) -> str:
    """
    Its ask to the user for your question and return the answer
    """
    try:

        try:
            from .agent.process import tts_if_you_can
            from .audio.record import quick_speech_to_text
        except:
            from agent.process import tts_if_you_can
            from audio.record import quick_speech_to_text

        print("TTS")
        tts_if_you_can(question, bypass_other_settings=True, not_threaded=True)
        print("TTS END")

        if wait_for_answer:
            return quick_speech_to_text(wait_for_answer)
        else:
            return quick_speech_to_text()
    except:
        traceback.print_exc()
        return False


def get_standard_tools():
    print("Tool len", len(_standard_tools_))
    last_list = [_standard_tools_[each] for each in _standard_tools_]
    return last_list



if __name__ == "__main__":
    print(ask_to_user("What is your age"))