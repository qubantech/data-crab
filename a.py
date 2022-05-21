import json

# encoding in UTF-8
unicodeData = {
    "string1": "ыыы",
    "string2": u"\u00f8"
}

print("utf-8")
encodedUnicode = json.dumps(unicodeData, ensure_ascii=False).encode('utf-8')
print("JSON ", encodedUnicode)

print("Decoding JSON", json.loads(encodedUnicode))
