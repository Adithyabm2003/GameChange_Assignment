import json
from playwright.sync_api import sync_playwright

def scrape_enbd_credit_cards():
    with sync_playwright() as p:
        # Launch browser (headless=True for background, False to watch it work)
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        
        print("🔍 Navigating to Emirates NBD...")
        url = "https://www.emiratesnbd.com/en/cards/credit-cards"
        
        # Wait until the network is quiet to ensure dynamic blocks load
        page.goto(url, wait_until="networkidle")
        
        # Wait for the specific container you provided
        try:
            page.wait_for_selector(".cc-block", timeout=10000)
        except:
            print("❌ Cards did not load. Check if the site is blocking the request.")
            browser.close()
            return []

        cards_data = []
        # Target the grid items
        card_containers = page.query_selector_all(".cardlist__grid-item")

        for card in card_containers:
            try:
                # 1. Get Title
                title_el = card.query_selector(".cc-block__title")
                name = title_el.inner_text().strip() if title_el else "Unknown Card"

                # 2. Get Benefits (List Items)
                benefit_elements = card.query_selector_all(".cc-block__content ul li")
                benefits = [li.inner_text().strip() for li in benefit_elements]

                # 3. Get 'Know More' Link
                link_el = card.query_selector("a.link-arrow")
                link = link_el.get_attribute("href") if link_el else ""
                if link.startswith("/"):
                    link = f"https://www.emiratesnbd.com{link}"

               

                cards_data.append({
                    "card_name": name,
                    "benefits": benefits,
                    "info_url": link,
                })
            except Exception as e:
                print(f"Skipping a card due to error: {e}")

        browser.close()
        return cards_data

# Run and save to JSON
scraped_cards = scrape_enbd_credit_cards()
with open("enbd_data.json", "w") as f:
    json.dump(scraped_cards, f, indent=4)

print(f"✅ Successfully captured {len(scraped_cards)} cards.")