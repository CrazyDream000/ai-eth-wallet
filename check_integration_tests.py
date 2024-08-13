import re
import json


def find_matches(file_path):
    used = []
    skipped = []
    # The regex pattern to match
    pattern = r'it(\.only)?\("([^"]+)"'

    try:
        # print(pattern)
        with open(file_path, "r") as file:
            for line in file:
                matches = re.findall(pattern, line)
                for match in matches:
                    used.append(match[1])
                    # print(match[1])

    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    # The regex pattern to match
    pattern = r'it\.skip\("([^"]+)"'

    try:
        # print(pattern)
        with open(file_path, "r") as file:
            for line in file:
                matches = re.findall(pattern, line)
                for match in matches:
                    skipped.append(match)
                    # print(match)

    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return used, skipped


# Example usage
file_path = "../walletapi/src/__tests__/integration/index.test.js"  # Replace with your file path
used1, skipped1 = find_matches(file_path)
file_path = "../walletapi/src/__tests__/condition/integration.test.js"  # Replace with your file path
used2, skipped2 = find_matches(file_path)
used = used1 + used2
skipped = skipped1 + skipped2

with open("part5.json", "r") as f:
    part5 = list(json.load(f).keys())
with open("part6.json", "r") as f:
    part6 = list(json.load(f).keys())
with open("part7.json", "r") as f:
    part7 = list(json.load(f).keys())
with open("part10.json", "r") as f:
    part10 = list(json.load(f).keys())
with open("replacements.json", "r") as f:
    replacements = json.load(f)
print(
    len(part5),
    "part5 tests",
    len(part6),
    "part6 tests",
    len(part7),
    "part7 tests",
    len(part10),
    "part10 tests",
    len(part5) + len(part6) + len(part7) + len(part10),
    "total",
)
res = part7 + part10
prev = used + skipped
ps = []
rs = []
for p in prev:
    if p not in res:
        ps.append(p)
        # print('p', p)
for p in res:
    if p not in prev:
        if p in replacements and replacements[p] in prev:
            continue
        else:
            rs.append(p)
        # print('r', p)
print(len(prev), "integration tests", len(res), "ai tests", len(ps), "extra", len(rs), "missing")
# with open("p7.txt", "w") as f:
# f.write('\n'.join(part7))
print("not in integration tests:")
count7 = 0
count10 = 0
for ix, r in enumerate(rs):
    if r in part7:
        count7 += 1
        print(count7, 'seven', r)
    elif r in part10:
        count10 += 1
        print(count10, 'nine', r)
    else:
        print('\n\nWHAT\n\n')
