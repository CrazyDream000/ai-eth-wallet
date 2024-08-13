# run test/trial2.py to get source code (needs edits to get abi as well)
# run solc --userdoc --devdoc <main_contract> > out.txt to get natspec
# run rgx.py to convert natspec to json
# run combine.py to combine abi and natspec into properly formatted docs
# append protocol docs to general endpoints and test

# NEW (UNDEVELOPED) METHOD
# after building 
{
  "language": "Solidity",
  "sources": {
    "aave/lib/aave-v3-core/contracts/protocol/pool/Pool.sol": {
      "urls": ["aave/lib/aave-v3-core/contracts/protocol/pool/Pool.sol"]
    },
    "aave/lib/aave-v3-core/contracts/protocol/libraries/helpers/Errors.sol": {
      "urls": ["aave/lib/aave-v3-core/contracts/protocol/libraries/helpers/Errors.sol"]
    }
  },
  "settings": {
    "outputSelection": {
      "*": {
        "Pool": [ "abi", "devdoc", "userdoc" ]
      }
    }
  }
}
# from return value of smart contract source code, run
# solc --standard-json input.json  | jq . > o.json
# then get abi, dev docs, and user docs from there and do similar parsing

# NOT GOOD
# - too large of a context
# - how to detect protocol, get correct address for action, rebuild possible functions?
# - too often not natspec documented