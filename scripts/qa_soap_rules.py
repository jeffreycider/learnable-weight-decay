"""Render the static mathematics page at desktop and phone widths."""
import asyncio
import functools
import http.server
import json
from pathlib import Path
import threading

from playwright.async_api import async_playwright

ROOT=Path(__file__).resolve().parents[1]/'sites'/'soap-update-rules'
OUT=ROOT/'qa';OUT.mkdir(exist_ok=True)

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self,*args):pass

server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(QuietHandler,directory=str(ROOT/'dist')))
threading.Thread(target=server.serve_forever,daemon=True).start()

async def main():
    async with async_playwright() as p:
        browser=await p.chromium.launch(headless=True,args=['--no-sandbox'])
        page=await browser.new_page()
        errors=[];requests=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('request',lambda r:requests.append(r.url))
        await page.goto(f'http://127.0.0.1:{server.server_port}/')
        await page.evaluate('document.fonts.ready')
        results=[]
        for label,width,height in [('desktop',1280,900),('phone',390,844),('small-phone',320,700)]:
            await page.set_viewport_size({'width':width,'height':height})
            await page.evaluate('scrollTo(0,0)')
            await page.screenshot(path=str(OUT/f'{label}-top.png'))
            checks=await page.evaluate('''() => ({
              width:innerWidth, documentWidth:document.documentElement.scrollWidth,
              equationCount:document.querySelectorAll('.equation').length,
              mathCount:document.querySelectorAll('.katex').length,
              mathErrors:document.querySelectorAll('.katex-error').length,
              missingAnchors:[...document.querySelectorAll('a[href^="#"]')]
                  .filter(a=>!document.getElementById(a.getAttribute('href').slice(1)))
                  .map(a=>a.getAttribute('href')),
              unavailableFonts:[...document.fonts].filter(f=>f.status==='error').length,
              rules:document.querySelectorAll('.rule').length,
              overflowingMath:[...document.querySelectorAll('.katex')].filter(e=>!e.closest('.equation,.table-wrap') && e.getBoundingClientRect().right>innerWidth).map(e=>({text:e.textContent.slice(0,90),right:e.getBoundingClientRect().right,parent:e.parentElement.outerHTML.slice(0,140)})),
              overflowLeaves:[...document.querySelectorAll('body *')].filter(e=>
                e.namespaceURI==='http://www.w3.org/1999/xhtml' && !e.closest('.equation,.table-wrap,.katex') && e.getBoundingClientRect().right>innerWidth+1
              ).slice(0,12).map(e=>({tag:e.tagName,cls:e.className,text:e.textContent.slice(0,90),right:e.getBoundingClientRect().right}))
            })''')
            results.append(dict(view=label,**checks))
            assert checks['documentWidth']==width,checks
            assert checks['mathErrors']==0 and not checks['missingAnchors'],checks
            assert checks['rules']==6 and checks['unavailableFonts']==0,checks
        await page.set_viewport_size({'width':390,'height':844})
        await page.locator('#r395').scroll_into_view_if_needed()
        await page.screenshot(path=str(OUT/'phone-pr395.png'))
        await page.locator('#weight').scroll_into_view_if_needed()
        await page.screenshot(path=str(OUT/'phone-weight-step.png'))
        await browser.close()
        assert not errors,errors
        assert all(u.startswith(f'http://127.0.0.1:{server.server_port}/') or u.startswith('data:') for u in requests),requests
        report=dict(viewports=results,page_errors=errors,network_requests=requests)
        (OUT/'checks.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))

try:asyncio.run(main())
finally:server.shutdown();server.server_close()
