from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from livekit.agents import function_tool, RunContext
import asyncio
import re
import shutil


def create_driver(headless: bool = True) -> webdriver.Chrome:
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service

    options = Options()
    chrome_binary = shutil.which("chromium") or shutil.which("chromium-browser") or shutil.which("google-chrome")
    if chrome_binary:
        options.binary_location = chrome_binary

    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    chromedriver = shutil.which("chromedriver")
    if not chromedriver:
        raise RuntimeError("chromedriver not found. Install chromium-driver in the agent image.")

    service = Service(chromedriver)
    return webdriver.Chrome(service=service, options=options)


def clean_html(raw_html: str, max_chars: int = 1500) -> str:
    """Strip non-interactive elements, return only actionable ones for the LLM."""
    soup = BeautifulSoup(raw_html, "html.parser")

    for tag in soup(["script", "style", "svg", "noscript",
                     "head", "meta", "link", "iframe", "img",
                     "footer", "nav", "aside"]):
        tag.decompose()

    kept = soup.find_all(["a", "button", "input", "select",
                          "textarea", "h1", "h2", "h3", "p", "li"])

    allowed_attrs = {"id", "class", "href", "type", "name",
                     "placeholder", "aria-label", "value", "role", "data-testid"}

    result = []
    for tag in kept:
        tag.attrs = {k: v for k, v in tag.attrs.items() if k in allowed_attrs}
        text = str(tag).strip()
        if text:
            result.append(text)

    return " ".join(result)[:max_chars]


class BrowserTools:
    driver: webdriver.Chrome = None
    _stop_requested: bool = False

    async def _ensure_driver(self):
        if hasattr(self, "wait_for_driver"):
            try:
                await self.wait_for_driver()
            except Exception:
                return "FAILED: Browser took too long to start."
        if self.driver is None:
            return "FAILED: Browser not initialized."
        return None

    async def emit_browser_event(self, event_type: str, message: str):
        pass  # overridden by VoiceAssistant — only used for start/done status

    @function_tool()
    async def get_current_url(self, context: RunContext) -> str:
        """Return the title and URL of the currently open browser tab."""
        err = await self._ensure_driver()
        if err:
            return err
        try:
            url   = await asyncio.to_thread(lambda: self.driver.current_url)
            title = await asyncio.to_thread(lambda: self.driver.title)
            return f"Current page: {title} | {url}"
        except Exception as e:
            return f"FAILED: {str(e)}"

    @function_tool()
    async def get_page_html(self, context: RunContext) -> str:
        """
        Return a compact snapshot of the current page's content and interactive elements.
        Always call this before clicking or typing to get correct selectors.
        """
        err = await self._ensure_driver()
        if err:
            return err
        try:
            await asyncio.sleep(1.0)
            html  = await asyncio.to_thread(lambda: self.driver.page_source)
            title = await asyncio.to_thread(lambda: self.driver.title)
            cleaned = clean_html(html)
            print(f"[get_page_html] {title} — {len(cleaned)} chars")
            return f"PAGE: {title}\n{cleaned}"
        except Exception as e:
            return f"FAILED: {str(e)}"

    @function_tool()
    async def navigate(self, context: RunContext, url: str, new_tab: bool = False) -> str:
        """Navigate the browser to a URL. Use new_tab=True to open in a new tab."""
        err = await self._ensure_driver()
        if err:
            return err
        try:
            domain = re.sub(r'https?://(www\.)?', '', url).split('/')[0].split('?')[0]
            print(f"[navigate] Opening {domain}")

            def _navigate():
                if new_tab:
                    self.driver.execute_script("window.open(arguments[0], '_blank');", url)
                    self.driver.switch_to.window(self.driver.window_handles[-1])
                else:
                    self.driver.get(url)

            await asyncio.to_thread(_navigate)
            await asyncio.sleep(2.5)
            title = await asyncio.to_thread(lambda: self.driver.title)
            print(f"[navigate] Loaded: {title}")
            return f"Navigated. Page title: {title}"
        except Exception as e:
            return f"FAILED: {str(e)}"

    @function_tool()
    async def click_element(self, context: RunContext, selector: str) -> str:
        """Click a page element by CSS selector or XPath. Always call get_page_html first."""
        err = await self._ensure_driver()
        if err:
            return err
        try:
            def _click():
                try:
                    el = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                except Exception:
                    el = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                label = el.text.strip() or "button"
                el.click()
                return label

            label = await asyncio.to_thread(_click)
            print(f"[click] {label[:60]}")
            return "Clicked"
        except Exception as e:
            return f"FAILED: {str(e)}"

    @function_tool()
    async def type_into_field(
        self,
        context: RunContext,
        selector: str,
        text: str,
        press_enter: bool = False
    ) -> str:
        """Type text into an input field by CSS selector. Set press_enter=True to submit."""
        err = await self._ensure_driver()
        if err:
            return err
        try:
            def _type():
                el = WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                el.clear()
                el.send_keys(text)
                if press_enter:
                    el.send_keys(Keys.RETURN)

            await asyncio.to_thread(_type)
            print(f"[type] '{text}'")
            return "Typed successfully"
        except Exception as e:
            return f"FAILED: {str(e)}"

    @function_tool()
    async def run_js(self, context: RunContext, script: str) -> str:
        """Execute JavaScript in the browser. Use only when click/type cannot handle it."""
        err = await self._ensure_driver()
        if err:
            return err
        try:
            result = await asyncio.to_thread(self.driver.execute_script, script)
            return str(result) if result is not None else "Done"
        except Exception as e:
            return f"FAILED: {str(e)}"

    @function_tool()
    async def stop_browser(self, context: RunContext) -> str:
        """Stop any ongoing browser task."""
        self._stop_requested = True
        return "Browser task stopped."

    @function_tool()
    async def switch_tab(self, context: RunContext, tab_index: int) -> str:
        """Switch to a browser tab by zero-based index."""
        err = await self._ensure_driver()
        if err:
            return err
        try:
            def _switch():
                handles = self.driver.window_handles
                if tab_index >= len(handles):
                    return f"Only {len(handles)} tabs open"
                self.driver.switch_to.window(handles[tab_index])
                return self.driver.title

            title = await asyncio.to_thread(_switch)
            return f"Switched to tab {tab_index}: {title}"
        except Exception as e:
            return f"FAILED: {str(e)}"
