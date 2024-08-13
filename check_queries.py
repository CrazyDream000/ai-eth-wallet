import os
from dotenv import load_dotenv
import psycopg
import json

load_dotenv()

db_config = {
    "dbname": os.getenv("DATABASE_NAME"),
    "user": os.getenv("DATABASE_USER"),
    "password": os.getenv("DATABASE_PASSWORD"),
    "host": os.getenv("DATABASE_HOST"),
    "port": int(os.getenv("DATABASE_PORT", 5432)),
}
blacklist = [
    None,
    "",
    "0x024cDB696a719f37B324a852085A68786D269212",
    "0x28129f5B8b689EdcB7B581654266976aD77C719B",
    "0x6F54510F06F3b9c95AD36d76F27CfbF57BEBe889",
    "0xd4129CAf7596B0B8e744608189Aee22184328447",
    "0xf0F6ef4bd30266eb5483E6Bd1e460903eADba0E6",
    "0x0666E6252a6bC3A4A186eD2e004643D7f2418b57",
    "0xbB3d4097E9F1279f07E981EAFF384Eb6566fbE2d",
    "0xa5Ef861278D7bf18A8A2068a01D66FBdeD93A1BD"
]
with psycopg.connect(**db_config) as conn:
    res = conn.execute(
        "SELECT DISTINCT ON (inputted_query) id, user_address, inputted_query, generated_api_calls, executed_status from tracking where created > 1716210058;",
    ).fetchall()
filtered = []
sorted_list = sorted(res, key=lambda x: x[0], reverse=True)
for r in sorted_list:
    if r[1] not in blacklist and 'niyant.eth' not in r[2]:
        filtered.append(r)
actions = {}
addresses = {}
with open("temp.txt", "w") as f:
    for r in filtered:
        f.write(f"{r[2]}\n")
        # print(r)
        i = 0
        word = ''
        while word == '':
            word = r[2].split(" ")[i].lower().replace("/","")
            i += 1
        actions[word] = actions.get(word, 0) + 1
        addresses[r[1]] = addresses.get(r[1], 0) + 1
print(dict(sorted(actions.items(), key=lambda item: item[1], reverse=True)))
print(dict(sorted(addresses.items(), key=lambda item: item[1], reverse=True)))
print(len(filtered))

# with open("test/newanswers.json", "r") as f:
    # data = json.load(f)
# 
# with open("temp.txt", "a") as f:
    # f.write("\n\n\n\n")
    # for d, v in list(data.items()):
        # f.write(f"{d}\n")
# 
# d = list(data.keys())
# for r in filtered:
    # if r[2] not in d and r[2].lower() not in d:
        # print(r[2])