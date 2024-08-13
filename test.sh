#!/bin/bash

# Check if two integers are provided as arguments
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <integer1> <integer2>"
  exit 1
fi

# Assign
integer0="$1"
integer1="$2"

# Calculate integer2 based on integer1
integer2=$((integer1 + 1))

# Create the JSON files test/res{integer1}, test/res{integer2}
echo '{}' > "test/res${integer1}.json"
echo '{}' > "test/res${integer2}.json"

# Call the Python script with the three integers as arguments
python3 main.py "$integer0" "$integer1"
python3 test/edit.py "$integer1" "$integer2"
python3 test/test.py "$integer1" "$integer2"
