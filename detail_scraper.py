import json
import requests
from bs4 import BeautifulSoup
import time
import pandas as pd

def scrape_card_details(json_file_path, output_file='scraped_cards_data.json'):
    # Load the URLs from your provided file
    with open(json_file_path, 'r') as f:
        cards = json.load(f)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.emiratesnbd.com/'
    }

    results = []

    print(f"Starting extraction for {len(cards)} cards...")

    for index, card in enumerate(cards):
        url = card.get('info_url')
        name = card.get('card_name')
        
        if not url:
            continue

        print(f"[{index + 1}/{len(cards)}] Scraping: {name}...")
        
        try:
            # Adding a small delay to avoid being blocked by the bank's security
            time.sleep(2) 
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extracting the main content area (standard for Emirates NBD card pages)
            # We look for common containers like 'main', 'article', or specific card divs
            content_sections = soup.find_all(['section', 'div'], class_=['card-details', 'product-details', 'rich-text'])
            
            full_text = ""
            if content_sections:
                for section in content_sections:
                    full_text += section.get_text(separator=' ', strip=True) + "\n"
            else:
                # Fallback to body text if specific containers aren't found
                full_text = soup.get_text(separator=' ', strip=True)

            # Structure the data
            card_detail = {
                "card_name": name,
                "url": url,
                "raw_content": full_text,
                "summary_benefits": card.get('benefits', [])
            }
            results.append(card_detail)

        except Exception as e:
            print(f"Error scraping {name}: {e}")
            results.append({
                "card_name": name,
                "url": url,
                "error": str(e)
            })

    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    # Also save to CSV for easier reading in Excel
    df = pd.DataFrame(results)
    df.to_csv('scraped_cards_data.csv', index=False)
    
    print(f"Successfully saved data to {output_file} and scraped_cards_data.csv")

# To run the script:
scrape_card_details('enbd_data.json')