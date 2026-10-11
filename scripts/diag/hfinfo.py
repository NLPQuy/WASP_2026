import json, urllib.request, sys
for m in sys.argv[1:]:
    try:
        d = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{m}", timeout=25))
        sz = sum(f.get("size") or 0 for f in d.get("siblings", []) if f["rfilename"].endswith((".safetensors",".bin")))
        print(f"{m:52s} gated={str(d.get('gated')):6s} pipe={str(d.get('pipeline_tag')):22s} dl={d.get('downloads',0)}")
    except Exception as e:
        print(f"{m:52s} LOI {e}")
