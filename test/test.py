import json
from collections import defaultdict
import sys

data = {}
if len(sys.argv) != 3:
    print("Usage: python script.py 70 71")
    sys.exit(1)

try:
    inputs = [int(arg) for arg in sys.argv[1:3]]
except ValueError:
    print("Invalid input. Please provide two integer values.")
    sys.exit(1)

for i in inputs:
    data[i]= {}
    with open(f"test/res{i}.json") as f:
        res = json.load(f)
    with open("test/answers2.json") as f:
        answers2 = json.load(f)
    with open("test/newanswers.json") as f:
        newanswers = json.load(f)

    error = 0
    for r, s in list(res.items()):
        if r.lower() in answers2:
            na = answers2[r.lower()]
        else:
            print('writing', r.lower())
            continue
            with open("test/newanswers.json", "w") as f:
                newanswers[r.lower()] = [s]
                json.dump(newanswers, f)
            na = [s]
        # fna = json.dumps(na)
        # fna = json.loads(fna)
        # if len(fna) > 0:
            # fna[0] = {key for key, value in fna[0].items() if value != "all"}
        # fs = json.dumps(s)
        # fs = json.loads(fs)
        # if len(fs) > 0:
            # fs[0] = {key for key, value in fs[0].items() if value != "all"}
        sna = []
        for nna in na:
            sna.append(sorted(nna, key=lambda x: x["name"]))
        ss = sorted(s, key=lambda x: x["name"])
        if s not in na and ss not in sna:
            error += 1
            data[i][r.lower()] = s
    l = len(res)
    # print(i, error, l, (l - error) / l)

list1 = list(data[inputs[0]].keys())
list2 = list(data[inputs[1]].keys())

from collections import Counter

# Combine all lists into one
combined_list = list1+list2

# Use Counter to count occurrences
item_counts = Counter(combined_list)

sorted_items = sorted(item_counts.items(), key=lambda x: x[1])

# Print the sorted counts
errored = []
for item, count in sorted_items:
    lists_with_item = [
        i + 1 for i, lst in enumerate([list1,list2]) if item in lst
    ]
    if lists_with_item == [1,2]:
        errored.append(item)

print(f"{len(errored)}/{l} wrong, accuracy is {(l - len(errored)) / l}")
for ix, e in enumerate(errored):
    print(ix+1, e)