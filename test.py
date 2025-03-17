import requests
from bs4 import BeautifulSoup

url = "https://docs.smith.langchain.com/tutorials/Developers/backtesting"
response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.content, "html.parser")
    text_content = soup.get_text()
    print(text_content)  # This will print the entire page text
else:
    print("Failed to retrieve content from the URL.")