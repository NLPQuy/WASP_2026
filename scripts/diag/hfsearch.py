import json, sys, urllib.request
def q(term, limit=8):
    url = f"https://huggingface.co/api/models?search={term}&sort=downloads&direction=-1&limit={limit}"
    try:
        d = json.load(urllib.request.urlopen(url, timeout=25))
    except Exception as e:
        print(f"### {term}: LOI {e}"); return
    print(f"### {term}")
    for m in d:
        print(f"  {m.get('downloads',0):>9}  {m['id']}")
for t in sys.argv[1:]:
    q(t)
