import asyncio
from pyppeteer import launch
from bs4 import BeautifulSoup
import requests

async def search(query):
    browser = await launch()
    page = await browser.newPage()
    await page.goto('https://www.google.com/')
    await page.type('input[name="q"]', query)
    await page.keyboard.press('Enter')
    await page.waitForNavigation()
    results = await page.evaluate('''() => {
        const links = Array.from(document.querySelectorAll('a'));
        return links.map(link => ({
            title: link.textContent,
            url: link.href
        }));
    }''')
    link = results[len(results)//2]
    return link

def get_result(query):
    return asyncio.get_event_loop().run_until_complete(search(query))
