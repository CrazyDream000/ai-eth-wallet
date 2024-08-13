import re
import json

# Read the text file
with open('test/out.txt', 'r') as f:
    file_content = f.read()

# Define the start and end patterns for the desired section
start_pattern = r'^======= test/aave/lib/aave-v3-core/contracts/protocol/pool/Pool.sol:Pool ======='
end_pattern = r'^======= '

# Find the start and end positions of the desired section
start_match = re.search(start_pattern, file_content, re.MULTILINE)
end_match = re.search(end_pattern, file_content[start_match.end():], re.MULTILINE)

if start_match and end_match:
    start_pos = start_match.end()
    end_pos = start_match.end() + end_match.start()

    # Extract the section content
    section_content = file_content[start_pos:end_pos].strip()
    x = section_content.split('Developer Documentation\n')[1]
    y = json.loads(x.split('User Documentation\n')[1])
    z = json.loads(x.split('User Documentation\n')[0])
    if y:
        print("FOUND")
        with open('test/outu.json', 'w') as f:
            json.dump(y, f)
    else:
        print("User documentation not found in the section.")
    if z:
        print("FOUND")
        with open('test/outd.json', 'w') as f:
            json.dump(z, f)
    else:
        print("Developer documentation not found in the section.")
else:
    print("Section not found in the file.")
