# Standard library imports
import smtplib
import sys
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from urllib.parse import urljoin

# Third-party imports
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException
from tqdm import tqdm
from colorama import Fore, Style, init
from dateutil.parser import parse

# Local application/library specific imports
import config # Import the configuration file

# --- Scraper Functions ---

def scrape_main_page(url, headers):
    """
    Scrapes the main page of The Economic Times markets section to find article headlines and links.
    """
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = []
        headline_links = soup.find_all('a', class_='font_faus')
        for link in headline_links:
            if link.has_attr('href') and "articleshow" in link['href']:
                headline_text = link.get_text(strip=True)
                article_url = urljoin(url, link['href'])
                articles.append({'headline': headline_text, 'url': article_url})
        return articles
    except RequestException as e:
        print(f"{Fore.RED}Error fetching the main page: {e}{Style.RESET_ALL}")
        return []
    except Exception as e:
        print(f"{Fore.RED}An unexpected error occurred on the main page: {e}{Style.RESET_ALL}")
        return []

def scrape_article_page(article_url, headers):
    """
    Scrapes an individual article page to extract the summary and publication date.
    """
    try:
        response = requests.get(article_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # --- Date Scraping Logic ---
        date_str = 'N/A'
        meta_tags = [
            {'property': 'article:published_time'},
            {'property': 'og:published_time'},
            {'name': 'publish-date'}
        ]
        for meta in meta_tags:
            date_tag = soup.find('meta', meta)
            if date_tag and date_tag.has_attr('content'):
                date_str = date_tag['content']
                break
        
        if date_str == 'N/A':
            script_tags = soup.find_all('script', type='text/javascript')
            for script in script_tags:
                if script.string:
                    match = re.search(r'"publishedDate"[:\s]+"(.*?)"', script.string)
                    if match:
                        date_str = match.group(1)
                        break

        if date_str == 'N/A':
            time_tag = soup.find('time', class_='jsdtTime')
            if time_tag and time_tag.has_attr('data-dt'):
                date_str = time_tag['data-dt']
            elif time_tag:
                date_str = time_tag.get_text(strip=True)

        # --- Date Normalization ---
        date = 'N/A'
        if date_str and date_str != 'N/A':
            try:
                if date_str.isdigit() and len(date_str) > 10:
                    date_obj = datetime.fromtimestamp(int(date_str) / 1000)
                else:
                    date_obj = parse(date_str, fuzzy=True, default=datetime(1, 1, 1))

                if date_obj.year > 1990:
                    date = date_obj.strftime('%Y-%m-%d')
                else:
                    date = 'N/A'
            except (ValueError, TypeError, OverflowError):
                date = 'N/A'

        # --- Summary Scraping ---
        summary_div = soup.find('div', class_='artText')
        summary = summary_div.get_text(strip=True, separator=' ') if summary_div else 'N/A'
        
        return {'summary': summary, 'date': date}
    except RequestException as e:
        print(f"\n{Fore.YELLOW}Could not fetch {article_url}: {e}{Style.RESET_ALL}")
        return None
    except Exception as e:
        print(f"\n{Fore.RED}An unexpected error occurred on {article_url}: {e}{Style.RESET_ALL}")
        return None

def get_latest_articles(limit=5):
    """
    Scrapes the latest financial news articles, returning them as a DataFrame.
    """
    init(autoreset=True)
    base_url = "https://economictimes.indiatimes.com/markets"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36'}
    finance_keywords = [
        'stock', 'market', 'finance', 'ipo', 'nse', 'bse', 'sensex', 'nifty', 'equity', 'shares', 'investing',
        'trading', 'investor', 'economy', 'mutual fund', 'bond', 'forex', 'commodity', 'earnings', 'profit',
        'revenue', 'bank', 'gdp', 'fed', 'treasury', 'currency', 'commodities', 'gold', 'silver', 'oil'
    ]
    
    articles = scrape_main_page(base_url, headers)
    scraped_data = []

    if not articles:
        print(f"{Fore.RED}No articles found on the main page.{Style.RESET_ALL}")
        return pd.DataFrame()

    print(f"\nFound {len(articles)} potential articles. Scraping and filtering for the latest {limit}...")
    
    with tqdm(total=limit, desc="Scraping Latest Articles", ncols=100) as pbar:
        for article in articles:
            if len(scraped_data) >= limit:
                break

            if any(keyword in article['headline'].lower() for keyword in finance_keywords):
                article_details = scrape_article_page(article['url'], headers)
                if article_details:
                    if any(keyword in article_details['summary'].lower() for keyword in finance_keywords):
                        full_article_data = {**article, **article_details}
                        scraped_data.append(full_article_data)
                        pbar.update(1)

    if not scraped_data:
        print(f"\n{Fore.YELLOW}No new articles matching the finance keywords were found.{Style.RESET_ALL}")
        return pd.DataFrame()
        
    return pd.DataFrame(scraped_data)

# --- Report Generator Function ---

def generate_html_report(articles_df):
    """
    Generates an HTML report from a DataFrame of financial articles.
    """
    df = articles_df.copy()

    # --- Data Cleaning and Sorting ---
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    current_year = datetime.now().year
    df.loc[df['date'].dt.year < 1990, 'date'] = pd.NaT
    df.loc[df['date'].dt.year > current_year, 'date'] = pd.NaT
    df.sort_values(by='date', ascending=False, inplace=True)
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    df['date'] = df['date'].fillna('Date Not Available')

    html_style = """
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; background-color: #f4f4f4; }
        h1, h3 { color: #2c3e50; }
        h1 { text-align: center; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        .article { background: #fff; border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .article h3 { margin-top: 0; }
        .article p { margin-bottom: 0; }
        .date { font-size: 0.9em; color: #7f8c8d; text-align: right; }
        footer { text-align: center; margin-top: 30px; font-size: 0.8em; color: #7f8c8d; }
    </style>
    """
    html_content = f"""
    <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Daily Financial Intelligence Report</title>{html_style}</head><body>
    <h1>Daily Financial Intelligence</h1>
    <p style="text-align:center;">Generated by Yash's AI Bot on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p><hr>
    """
    for _, row in df.iterrows():
        html_content += f"""
        <div class="article"><h3>{row['headline']}</h3><p class="date">{row['date']}</p><p>{row['summary']}</p></div>
        """
    html_content += """
    <footer><p>&copy; 2026 Yash's AI Bot. All rights reserved.</p></footer></body></html>
    """
    return html_content

# --- Email Function ---

def send_email(html_content, sender_email, sender_password, recipient_email):
    """
    Constructs and sends an email with the given HTML content.
    """
    if not all([sender_email, sender_password, recipient_email]):
        print("Error: Please make sure SENDER_EMAIL, APP_PASSWORD, and RECEIVER_EMAIL are set in config.py", file=sys.stderr)
        sys.exit(1)

    message = MIMEMultipart("alternative")
    today_date = datetime.now().strftime('%Y-%m-%d')
    message["Subject"] = f"Daily Financial Intelligence Report - {today_date}"
    message["From"] = sender_email
    message["To"] = recipient_email
    message.attach(MIMEText(html_content, "html"))

    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
    print(f"Connecting to SMTP server at {SMTP_SERVER}:{SMTP_PORT}...")

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            print("Logging in...")
            server.login(sender_email, sender_password)
            print("Sending email...")
            server.sendmail(sender_email, recipient_email, message.as_string())
        print(f"Email sent successfully to {recipient_email}!")
    except smtplib.SMTPAuthenticationError:
        print("Error: SMTP authentication failed. Please check your email/password or 'App Password' settings.", file=sys.stderr)
        sys.exit(1)
    except smtplib.SMTPException as e:
        print(f"An SMTP error occurred: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

# --- Main Execution Block ---

if __name__ == "__main__":
    print("--- Starting News Automation Process ---")
    
    # 1. Scrape the latest 5 articles
    print("Fetching latest articles...")
    articles_df = get_latest_articles(limit=5)
    
    if not articles_df.empty:
        # 2. Generate the HTML report from the DataFrame
        print("Generating HTML report...")
        report_content = generate_html_report(articles_df)
        
        if report_content:
            # 3. Save the report to a dynamically named HTML file
            today_date_str = datetime.now().strftime('%Y-%m-%d')
            output_filename = f"{today_date_str}.html"
            try:
                with open(output_filename, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                print(f"Report successfully saved to {output_filename}")
            except IOError as e:
                print(f"Error: Could not save report to file '{output_filename}': {e}", file=sys.stderr)

            # 4. Send the email
            send_email(
                html_content=report_content,
                sender_email=config.SENDER_EMAIL,
                sender_password=config.APP_PASSWORD,
                recipient_email=config.RECEIVER_EMAIL
            )
        else:
            print("Failed to generate HTML report.", file=sys.stderr)
    else:
        print("No articles found to generate a report.", file=sys.stderr)

    print("--- News Automation Process Finished ---")

    # Optional Scheduling Note:
    # To run this script daily, use your system's scheduler:
    # - Linux/macOS: Use 'cron'. Example: 0 8 * * * /usr/bin/python3 /path/to/your/news_automation.py
    # - Windows: Use 'Task Scheduler'.
