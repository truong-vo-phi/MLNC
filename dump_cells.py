with open('/marimo/notebook.py', encoding='utf-8') as f:
    text = f.read()
lines = text.splitlines()
print("Lines 550 to end:")
for idx in range(550, len(lines)):
    print(f"{idx+1}: {lines[idx]}")
