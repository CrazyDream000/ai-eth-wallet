import json

# Load the first JSON data
with open('test/outu.json', 'r') as f:
    user_data = json.load(f)['methods']
with open('test/outd.json', 'r') as f:
    dev_data = json.load(f)['methods']

# Load the second JSON data
with open('test/testabi.json', 'r') as f:
    output_data = json.load(f)

# Update the "description" field of each function in the output data
final = []
for obj in output_data:
    for func_name, info in user_data.items():
        if obj['name'] in func_name:
            obj['description'] = info['notice']
            final.append(obj)
            break
    else:
        final.append(obj)

output = []
for obj in final:
    for func_name, info in dev_data.items():
        if obj['name'] in func_name:
            if 'details' in info:
                obj['description'] += ". " + info['details']
            for act_name in list(obj['parameters']['properties'].keys()):
                if act_name in info['params']:
                    obj['parameters']['properties'][act_name]['description'] = info['params'][act_name]
            output.append(obj)
            break
    else:
        output.append(obj)
# Save the updated output data
with open('test/updated_output_data.json', 'w') as f:
    json.dump(final, f)
