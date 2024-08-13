import json

with open('test/newanswers.json') as f:
    glob = json.load(f)
with open('part5.json') as f:
    five = json.load(f)
with open('part6.json') as f:
    six = json.load(f)
with open('part7.json') as f:
    seven = json.load(f)
with open('part10.json') as f:
    ten = json.load(f)
with open('part11.json') as f:
    eleven = json.load(f)
with open('part12.json') as f:
    twelve = json.load(f)
with open('part13.json') as f:
    thirteen = json.load(f)
with open('part14.json') as f:
    fourteen = json.load(f)
with open('part15.json') as f:
    fifteen = json.load(f)
check = {}
for f, v in list(five.items()):
    if f in check:
        print('dup5', f)
    else:
        check[f] = v
print(len(check))
for f, v in list(six.items()):
    if f in check:
        print('dup6', f)
    else:
        check[f] = v
print(len(check))
for f, v in list(seven.items()):
    if f in check:
        print('dup7', f)
    else:
        check[f] = v
print(len(check))
for f, v in list(ten.items()):
    if f in check:
        print('dup10', f)
    else:
        check[f] = v
print(len(check))
for f, v in list(eleven.items()):
    if f in check:
        print('dup11', f)
    else:
        check[f] = v
print(len(check))
for f, v in list(twelve.items()):
    if f in check:
        print('dup12', f)
    else:
        check[f] = v
print(len(check))
for f, v in list(thirteen.items()):
    if f in check:
        print('dup13', f)
    else:
        check[f] = v
print(len(check))
for f, v in list(fourteen.items()):
    if f in check:
        print('dup14', f)
    else:
        check[f] = v
print(len(check))
for f, v in list(fifteen.items()):
    if f in check:
        print('dup15', f)
    else:
        check[f] = v
print(len(check))
print(len(glob))
add = []
update = []
cs = []
for g, v in list(glob.items()):
    names = [y['name'] for x in v for y in x]
    if "chat" in names or "support" in names:
        continue
    if g not in check:
        print('g1', g) 
        add.append(g)
    else:
        if check[g] != v:
            print('g2', g, v, '\n')
            update.append(g)
for c, v in list(check.items()):
    names = [y['name'] for x in v for y in x]
    if "chat" in names or "support" in names:
        continue
    if c not in glob:
        print('c1', c) 
        print("\n\nBAD\n\n")
        cs.append(c)
    else:
        if glob[c] != v and c not in update:
            print('c2', c, v, '\n')
            cs.append(c)
if cs == []:
    for a in add:
        fifteen[a] = glob[a]
    for u in update:
        if u in five:
            five[u] = glob[u]
        elif u in six:
            six[u] = glob[u]
        elif u in seven:
            seven[u] = glob[u]
        elif u in ten:
            ten[u] = glob[u]
        elif u in eleven:
            eleven[u] = glob[u]
        elif u in twelve:
            twelve[u] = glob[u]
        elif u in thirteen:
            thirteen[u] = glob[u]
        elif u in fourteen:
            fourteen[u] = glob[u]
        else:
            fifteen[u] = glob[u]

    with open('part5.json', 'w') as f:
        json.dump(five, f)
    with open('part6.json', 'w') as f:
        json.dump(six, f)
    with open('part7.json', 'w') as f:
        json.dump(seven, f)
    with open('part10.json', 'w') as f:
        json.dump(ten, f)
    with open('part11.json', 'w') as f:
        json.dump(eleven, f)
    with open('part12.json', 'w') as f:
        json.dump(twelve, f)
    with open('part13.json', 'w') as f:
        json.dump(thirteen, f)
    with open('part14.json', 'w') as f:
        json.dump(fourteen, f)
    with open('part15.json', 'w') as f:
        json.dump(fifteen, f)
        