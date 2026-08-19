from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://baloto.com/resultados', wait_until='domcontentloaded')
    print('URL=', page.url)
    print('TITLE=', page.title())
    print('TABLE_COUNT=', page.locator('table').count())
    for i in range(min(page.locator('table').count(), 10)):
        cls = page.locator('table').nth(i).get_attribute('class')
        print('TABLE_CLASS_', i, repr(cls))

    print('A_COUNT=', page.locator('a').count())
    for i in range(min(page.locator('a').count(), 60)):
        a = page.locator('a').nth(i)
        txt = a.inner_text().strip()
        href = a.get_attribute('href')
        cls = a.get_attribute('class')
        if txt or href or cls:
            print('A_', i, repr(txt), repr(href), repr(cls))

    print('HTML_SNIP_START')
    print(page.content()[:5000])
    print('HTML_SNIP_END')
    browser.close()
