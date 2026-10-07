"""Figure 6: screenshot of the web prototype classifying a BLB image without a temperature readout (BLB/thermalimage52a.jpg; chosen 7 Oct 2026).
How it was made (7 Oct 2026): the files of 02_Web_App_HuggingFace/ (identical to the live Space) were served locally
with `python3 -m http.server 8765`, with the TensorFlow.js 4.22.0 script taken from npm instead of the CDN; then
`python fig6_screenshot.py <image.jpg> <out.png>`, and the empty margin was trimmed. The time shown in the app (9.4 s)
reflects the software-rendered graphics of that machine; browser timings on an Apple M3 are in web_timing_M3.json."""
import sys, asyncio
from playwright.async_api import async_playwright
img, out = sys.argv[1], sys.argv[2]
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(args=['--use-angle=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist'])
        pg = await b.new_page(viewport={'width':900,'height':640}, device_scale_factor=2.5)
        await pg.goto('http://127.0.0.1:8765/index.html')
        await pg.wait_for_function("document.getElementById('status').textContent.includes('Model ready')", timeout=600000)
        print('backend', await pg.evaluate('tf.getBackend()'))
        await pg.set_input_files('#file', img)
        await pg.wait_for_function("document.getElementById('status').textContent.includes('Prediction')", timeout=600000)
        await pg.wait_for_timeout(800)   # let the bar transition (0.3 s) finish
        print(await pg.evaluate("document.getElementById('status').textContent"))
        print(await pg.evaluate("[...Array(6).keys()].map(i=>document.getElementById('n'+i).textContent+' '+document.getElementById('v'+i).textContent).join(' | ')"))
        await pg.locator('main').screenshot(path=out)
        await b.close()
asyncio.run(main())
