import requests
from bs4 import BeautifulSoup


def fetch_website_text(url: str) -> str:
    """
    Fetches and extracts the main text content from a web page.
    Removes scripts and styles, and returns clean text.
    Args:
        url (str): The URL of the web page to fetch.
    Returns:
        str: The extracted text content.
    Raises:
        requests.HTTPError: If the request fails.
    """
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    # Remove scripts and styles
    for script in soup(["script", "style"]):
        script.decompose()
    # Extract text
    text = soup.get_text(separator=' ', strip=True)
    return text
