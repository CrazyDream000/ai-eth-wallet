#!/bin/bash

# Create the JSON files
rm test/res99.json
rm test/res100.json
echo '{}' > "test/res99.json"
echo '{}' > "test/res100.json"

# Call the Python script with the three integers as arguments
python3 main.py split
python3 main.py runtest
python3 test/edit.py 99 100
python3 test/test.py 99 100
