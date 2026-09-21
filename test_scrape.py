from curl_cffi import requests
from bs4 import BeautifulSoup
import re
import json

url = 'https://www.webnovel.com/book/young-sheldon-twin-brother-of-georgie_36195261200123505/ch-1_97176996018182183'
s = requests.Session(impersonate='chrome')
r = s.get(url)

idx = r.text.find('var chapInfo=')
if idx != -1:
    snippet = r.text[idx:idx+15000]
    m_idx = re.search(r'"chapterIndex"\s*:\s*(\d+)', snippet)
    m_name = re.search(r'"chapterName"\s*:\s*"(.*?)"', snippet)
    m_next = re.search(r'"nextChapterId"\s*:\s*"(.*?)"', snippet)
    m_total = re.search(r'"totalChapterNum"\s*:\s*(\d+)', snippet)
    print('Index:', m_idx.group(1) if m_idx else 'None')
    print('Name:', m_name.group(1) if m_name else 'None')
    print('Next ID:', m_next.group(1) if m_next else 'None')
    print('Total Chapters:', m_total.group(1) if m_total else 'None')

soup = BeautifulSoup(r.text, 'html.parser')
# Extract title & paragraphs
paras = []
for p in soup.find_all('p'):
    txt = p.get_text(strip=True)
    # Exclude system/footer webnovel text if any
    if txt and not txt.startswith('©') and not 'WebNovel' in txt and not 'App Store' in txt:
        paras.append(txt)

print(f"Extracted {len(paras)} paragraphs.")
print("First 3 paragraphs:")
for p in paras[:3]:
    print("  -", p[:80])
