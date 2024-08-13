import json
import os

import openai
import requests
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def transfer(**kwargs):
    return "/transfer has been called successfully"
def swap(**kwargs):
    return json.dumps({"outputAmount": "outputAmount"})
def bridge(**kwargs):
    return json.dumps({"outputAmount": "outputAmount", "outputToken": "outputToken"})
def deposit(**kwargs):
    return json.dumps({"outputAmount": "outputAmount", "outputToken": "outputToken"})
def withdraw(**kwargs):
    return json.dumps({"outputAmount": "outputAmount", "outputToken": "outputToken"})
def claim(**kwargs):
    return json.dumps({"outputAmount": "outputAmount", "outputToken": "outputToken"})
def borrow(**kwargs):
    return json.dumps({"outputAmount": "outputAmount"})
def lend(**kwargs):
    return json.dumps({"outputAmount": "outputAmount", "outputToken": "outputToken"})
def repay(**kwargs):
    return json.dumps({"outputAmount": "outputAmount"})
def stake(**kwargs):
    return json.dumps({"outputAmount": "outputAmount", "outputToken": "outputToken"})
def unstake(**kwargs):
    return json.dumps({"outputAmount": "outputAmount"})
def harvest(**kwargs):
    return json.dumps({"outputAmount": "outputAmount", "outputToken": "outputToken"})
def long(**kwargs):
    return "/long has been called successfully"
def short(**kwargs):
    return "/short has been called successfully"
def lock(**kwargs):
    return json.dumps({"outputAmount": "outputAmount", "outputToken": "outputToken"})
def unlock(**kwargs):
    return json.dumps({"outputAmount": "outputAmount"})
def vote(**kwargs):
    return "/vote has been called successfully"
def condition(**kwargs):
    return "/condition has been called successfully"
def tim(**kwargs):
    return "/time has been called successfully"
    
def run_conversation(message):
    results = []
    behavior = """
    Completely and accurately satisfy the user query.
    Return the steps needed to schedule, trigger, and/or perform the user's intent.
    "all" is a valid input amount.
    "half" is a valid input amount.
    Only use the functions you have been provided with.
    Only use function inputs you have been provided with.
    Respond with a single sentence.
    Every function call should be unique.
    """
    behavior = [
        {"role": "system", "content": behavior},
    ]
    query = [
        {"role": "user", "content": message}
    ]
    messages = behavior + query
    with open('test/functions8.json', 'r') as file:
        functions = json.load(file)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        temperature=0,
        messages=messages,
        functions=functions,
        function_call="auto",
    )
    response_message = response["choices"][0]["message"]
    while response_message.get("function_call") and not response_message.get("content"):
        # print(response_message)
        results.append(response_message['function_call'])
        available_functions = {
            "transfer": transfer,
            "swap": swap,
            "bridge": bridge,
            "deposit": deposit,
            "withdraw": withdraw,
            "claim": claim,
            "borrow": borrow,
            "lend": lend,
            "repay": repay,
            "stake": stake,
            "unstake": unstake,
            "harvest": harvest,
            "long": long,
            "short": short,
            "lock": lock,
            "unlock": unlock,
            "vote": vote,
            "condition": condition,
            "time": tim
        }
        function_name = response_message["function_call"]["name"]
        function_to_call = available_functions[function_name]
        try:
            function_args = json.loads(response_message["function_call"]["arguments"])
        except Exception as e:
            print(response_message)
            raise e
        function_response = function_to_call(**function_args)
        messages.append(response_message)
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            temperature=0,
            messages=messages,
            functions=functions,
            function_call="auto",
        ) 
        response_message = response["choices"][0]["message"]
    messages.append(response_message)
    return results, response_message

def perform_message(message):
    results, response = run_conversation(message)
    print(message)
    print('Human Response:')
    print(response['content'])
    print('Transactions to Sign:')
    signs = []
    for data in results:
        name = data['name']
        args = json.loads(data['arguments'])
        print(name, args)
        signs.append({"name": name, "args": args})
    print('')
    # with open('test/res7.json', 'r') as f:
        # sofar = json.load(f)
    # sofar[message] = signs
    # with open('test/res7.json', 'w') as f:
        # json.dump(sofar, f)
    return results, response


def suite(step=0):
    if step == 0 or step == 1:
        perform_message("Swap 1 ETH for USDC")
        perform_message("Swap 1 ETH for USDC on Uniswap")
        perform_message("Bridge 1 USDT from Base to Arbitrum")
        perform_message("Bridge 1 USDT from Base to Arbitrum on Hop Protocol")
        perform_message("Transfer 10 DAI to niyant.eth")
        perform_message("Swap 1 ETH for USDC on Ethereum then bridge to Arbitrum")
        perform_message("Bridge 1 ETH from Ethereum to Optimism then buy USDC")
        perform_message("Bridge 1 WETH from Base to Ethereum and deposit in Aave")
    if step == 0 or step == 2:
        perform_message("Swap 10 ETH for USDC when gas is below 20")
        perform_message("Swap 10 ETH for USDC when ETH is below 1600")
        perform_message("Swap 10 ETH for USDC in twelve hours")
        perform_message("Swap 10 ETH for USDC at 5pm")
        perform_message("Swap 10 ETH for USDC in twelve hours, repeating every twelve hours")
        perform_message("Swap 10 ETH for USDC at 5pm, repeating every 1 hour")
    if step == 0 or step == 3:
        perform_message("Deposit all my WETH into Aave")
        perform_message("Swap all my WETH into USDC")
        perform_message("Buy USDT with all my WETH")
        perform_message("Bridge all my WETH to Base")
        perform_message("Withdraw 0.1 ETH from Compound and buy OP")
        perform_message("Bridge 3 ETH to Avalanche and buy OHM")
        perform_message("Use 3 ETH to buy OHM on Avalanche")
        perform_message("Buy GRAIL with 4 WETH")
        perform_message("Bridge all my tokens on Canto to Ethereum")
    if step == 0 or step == 4:
        perform_message("Open a short trade on Kwenta on BTC with 3 ETH with 3x leverage") # "Hey can you get some eth, convert it, deposit in kwenta, open a short trade with this leverage"
        perform_message("Withdraw from all my positions, convert to WETH, and bridge to Arbitrum")
        perform_message("swap eth for usdt, swap usdc for usdt, bridge usdt to arbitrum")
        perform_message("When gas is below 10, deposit 100 USDC into Morpho")
        perform_message("At 10am tomorrow, transfer 200 USDC to 0x2B605C2a76EE3F08a48b4b4a9d7D4dAD3Ed46bf3")
        perform_message("Stake 10 ETH on Rocket Pool")
        perform_message("Harvest all my positions on Arbitrum")
        perform_message("Swap all my tokens on Optimism to WETH and bridge to Arbitrum")
    if step == 0 or step == 5:
        perform_message("Swap 1 ETH to USDC, bridge to Arbitrum, deposit into JonesDAO, then deposit LP into Rodeo")
        perform_message("Bridge 1 ETH to Base, swap half to USDC, deposit into Kyber eth-usdc pool")
        perform_message("Harvest my MMF yield farms and automatically stake MMF every day at 8am")
        perform_message("Harvest my positions every Wednesday") # FAIL
        perform_message("3x leverage long GLP with 1000 USDC on GMX and swap 1000 USDC into UNI") # "Ok i want to 3x leverage GLP and zap out of USDC into some other asset thats support on Dolomite"
        perform_message("Swap 500 DAI for WBTC every day for a month when gas is less than 30") # perform_message("carry out x no of swap on the dapp daily for 1 month when gas is less than 30")
        perform_message("Claim and restake rewards from all my positions every Monday") # perform_message("Claim LP staking rewards/airdrops to add back to the LP")
        perform_message("Bridge 200 USDT from Ethereum to Base and buy PEPE") # perform_message("Bridge from mainnet and long a coin in one sweep.")
    if step == 0 or step == 6:
        perform_message("harvesting on camelot")
        perform_message("Using 2 ETH buy USDC, USDT, and DAI, then deposit into Curve tricrypto pool") # perform_message("buy x amount usdc , x amount usdt, x amount DAI, and stake in trycrypto curve pool")
        perform_message("Vote on my THENA position every week on Wednesday")
        perform_message("Deposit 100 ARB into Plutus, stake LP for PLS, then lock PLS") # perform_message("Deposit 100 ARB into Plutus and stake LP for PLS, then lock PLS")
        perform_message("withdraw position from trader joe")
        perform_message("Vote on Solidly every Wednesday and claim Solidly rewards every Thursday") # perform_message("voting solidly forks every week and claiming rewards of the same next day")
    if step == 0 or step == 7:
        perform_message("Borrow USDC from Compound and deposit into Aave") # FAIL
        perform_message("Borrow 1000 USDC from Compound and deposit into Aave")
        perform_message("Withdraw from all my positions on Ethereum and convert everything to ETH")
        perform_message("Vote, harvest, and restake all my positions every day")
        perform_message("Vote on all my positions every Sunday") # perform_message("Vote on all my positions once a week")
        perform_message("vote on the most optimal pair on solidly every wednesday at this time")
        perform_message("Harvest and restake all my positions every week")
    if step == 0 or step == 8:
        perform_message("Process rewards on Redacted Cartel, swap to WETH, and deposit into Blur, biweekly") # "process RLBTRFLY rewards bi weekly....then take the weth i receive and deposit into blur vault."
        perform_message("grab weekly rewards from ve(3,3) DEXes and convert them to ETH") # FAIL
        perform_message("Grab rewards from Balancer and convert to ETH every week") # FAIL
        perform_message("Bridge 1000 USDC from Ethereum to zkSync and deposit into PancakeSwap") # "spot a farm on defi lama and realize you dont have funds on lets say zkSync"
        perform_message("Withdraw 100 USDC from JonesDAO, bridge to Ethereum, and deposit it into Yearn") # "rebalancing pools and farms on different chains"
        perform_message("Claim and redeposit rewards on all my protocols every week on Wednesday") # "There are many different markets to claim rewards and reinvest as well for LP positions"
    if step == 0 or step == 9:
        perform_message("Buy BTC with 1 ETH every week") # FAIL # "Want to DCA every week"
        perform_message("Buy BTC with 1 ETH when BTC is at or below $25000 and sell 0.2 BTC for ETH when BTC is at or above $30000, forever") # "straddling buy / sell at specific prices" # "on Bitcoin, Buy at x price, Sell at y, Rinse and repeat for 48 hours" perform_message("arbitrage bot: buy btc on x and sell on y until price equilizes")
        perform_message("Claim and restake my Chronos position every week on Monday")
        perform_message("Bridge 4 USDT to Base")
        perform_message("Swap 3 ETH to USDC and deposit into the ETH-USDC pool on Dolomite")
        perform_message("Open a 2x ETH long on GMX with 1000 USDC")
        perform_message("Vote on my Thena position every Wednesday")
    if step == 0 or step == 10:
        perform_message("Withdraw 2 ETH from my ETH-USDC pool position on Camelot")
        perform_message("Claim STG from my Stargate positions, swap to WETH, and deposit back into Stargate")  # FAIL # "Compound my Stargate position"
        perform_message("for my position in pendle, if it reaches $1.50, sell it for usdc. Buy back with usdc at $1.20")  # FAIL # perform_message("for my position in pendle, if it reaches $1.50, sell it. Buy back at $1.20")
        perform_message("Stake my ARB on Arbitrum") # "I want to stake my arb, please give me 3 options"
        perform_message("Harvest my Balancer position and stake the rewards")
        perform_message("Withdraw half the liquidity from my Dolomite USDC-USDT position")
    if step == 0 or step == 11:
        perform_message("Claim wETH-GRAIL LP rewards from Camelot and sell for USDC")  # FAIL # "You get staking rewards as stables and wETH-GRAIL LP.; You gotta exit the LP then sell each individually for whatever you want"
        perform_message("Sell all of my tokens under $100 and convert to USDT on mainnet") # FAIL
        perform_message("sell all my usdc for eth if usdc goes below $.95") # "conditional triggers (e.g. depeg on a stable)"
        perform_message("Buy JONES with half my ETH, deposit into the ETH-JONES pool on Sushi, then trade LP for plsJones") # "take eth and buy Jones then pair into lp position on sushi then take the lp token and trade it for plsjones"
    if step == 0 or step == 12:
        perform_message("Buy ETH with 1000 USDC on Uniswap on Ethereum, bridge to Optimism, and sell for USDC on Velodrome") # perform_message("faster arbitrage across chains")
        perform_message("Swap 5000 USDC for ETH on Sushiswap on Ethereum, bridge to Base, sell ETH for USDC on KyberSwap, bridge USDC back to mainnet")  # FAIL # perform_message("arbitrage process and he had to bridge + swap + send it everywhere + go to velodrome")
        perform_message("buy wbtc with eth on uniswap and sell it for eth on sushiswap")
        perform_message("buy wbtc with 1 eth on uniswap")
        perform_message("swap 1 eth for usdt")
        perform_message("swap XYZ for ABC on pancakeswap in 35 minutes")
        perform_message("swap XYZ for ABC on pancakeswap at 11 PM UST") # FAIL
    if step == 0 or step == 13:
        perform_message("Claim my Camelot rewards, swap to USDC, and deposit back into Camelot") # perform_message("claiming rewards and compounding g into the pool")
        perform_message("Buy WBTC with 1 ETH every Sunday") # perform_message("setting up DCA buying based on time and buy/sells on price levels")
        perform_message("Withdraw from my Lodestar position") # perform_message("withdrawing from LPs/staking")
        perform_message("Harvest all my rewards on Arbitrum and buy ETH") # perform_message("harvesting rewards, but seeking them and twapping into new tokens I want to accumulate")
    if step == 0 or step == 14:
        perform_message("Lend 5 ETH, borrow 100 PT, then deposit 100 PT and 100 GLP into the PT-GLP pool on Pendle") # perform_message("PT-GLP and money markets for PT-GLP; Something were lacking is looping strategies on PT; Would love to set up a prompt and have users execute operation in one go")
        perform_message("Deposit ETH into Pendle when APY is 10%")  # FAIL # perform_message("Creating limit orders with pendle - by PT when yield is at a specific level")
        perform_message("Lend 250 SMP and borrow 125 LMP on Pendle") # perform_message("Will short the short maturity and long the long maturity; Hes arbing the yield between the two pools")
        # perform_message("Buy PT with USDC on Pendle, then loop PT on Dolomite 5 times") # perform_message("Wants to loop with pendle to earn yield; have to go pendle to buy PT, then go to a market to borrow PT, then go back to pendle to buy PT")
    if step == 0 or step == 15:
        perform_message("Claim rewards from Camelot, swap rewards and GRAIL into xGRAIL, then deposit xGRAIL into Camelot") # FAIL # perform_message("claim rewards from camelot, 3 transactions to claim, plus two additional transactions to convert dust grail into xgrail then allocate to dividend")
        perform_message("Claim Redacted rewards and relock BTRFLY") # perform_message("relocked RL BTRFLY, CLAIM rewards also")
    if step == 0 or step == 16:
        perform_message("Withdraw all my USDC from Aave and deposit into Compound") # perform_message("Withdraw usdc from aave if compound usdc interest rate > aave.")
        perform_message("If bitcoin goes below 15k, buy eth") # perform_message("If bitcoin hits 15k, buy eth") # perform_message("marketbuy eth if bitcoin touches 15k")
        perform_message("Claim Stargate rewards, swap to ETH, redeposit") # perform_message("autocompounding any position")
        perform_message("Buy ETH with 5000 USDC. Sell ETH for USDC if the price goes below 1000 or above 3000")  # FAIL # perform_message("Buy ETH with 5000 USDC. Sell if ETH hits 1000 or 3000") # perform_message("setting up take profit/stop loss or optimizing pools")
    if step == 0 or step == 17:
        perform_message("Buy DPX with RDPX if the price of DPX/RDPX <= 0.8")  # FAIL # perform_message("Swing trading non-pooled pairs based on their ratio (dpx/rdpx)")
        perform_message("Unstake all my plsDPX and sell it for DPX if the price of plsDPX/DPX < 0.95") # FAIL # perform_message("Unstaking and selling when ratio between a liquid derivative and the native asset hits certain ratio, being able to reverse that operation (say plsdpx on Plutus)")
        perform_message("Deposit 1000 USDC and borrow ARB on Paraspace, bridge from Arbitrum to Ethereum, deposit ARB and borrow USDC on Compound, deposit USDC on GMD Protocol") # perform_message("Go to ParaSpace on Arbitrum deposit collateral ($ETH or $USDC) apy 3.9% and 2.7%, Take a loan from $ARB apy 1%, Go to compound and deposit $ARB and borrow $USDC apy 2.3%, Go to @GMDprotocol and deposit $USDC apy 7%")
        perform_message("Bridge 4 ETH from Arbitrum to Base and buy COIN when gas is under 12") # perform_message("Bridge from Arbitrum to Base and buy COIN when gas is under 12")
    if step == 0 or step == 18:
        perform_message("Swap all my tokens to ETH and buy ARB when gas is below 10") # perform_message("consolidate entire portfolio into ETH and get it onto arb when gas is low")
        perform_message("Swap all my tokens to ETH and transfer to niyant.eth on mainnet") # perform_message("turn everything into eth and send to preset addy on main.")
        perform_message("Swap half of all my tokens to ETH and transfer to niyant.eth on mainnet")
        perform_message("can you use my DAI to purchase sWeed")
        perform_message("Use DAI to purchase sWeed")
    if step == 0 or step == 19:    
        perform_message("Withdraw all my USDC and USDT from Rodeo, convert to ETH, and bridge all of it to mainnet") # FAIL # "Withdraw from Rodeo, convert to ETH, and bridge all of it to mainnet"
        perform_message("Deposit 50 USDC and 50 USDT into DODO Finance USDC-USDT pool, then every Friday claim DODO and swap to usdt") # FAIL # "You LP USDC-USDT and earn DODO at 7-8% APY which you can dump for stables"
        perform_message("when my ETH balance hits 1, buy 0.5 ETH worth of SAINT once the price of SAINT/ETH is under 20 and gas under 15") # FAIL 
        perform_message("Once my Plutus rewards hit 2 ETH, claim rewards and transfer to person.eth") # FAIL 
        perform_message("Stake STG on stargate, then every Friday claim and restake rewards") # perform_message("Stake STG on stargate, every Friday claim and restake rewards every week on Friday")



def antisuite():
    # working just not integrated
    perform_message("get money from blur, send to arb, open up position on NFT perp all in one go")
    perform_message("I want to stake my arb, please give me 3 options")
    perform_message("What I do with my stETH?")
    
    # not working
    perform_message("the idea is youre minting DPXETH, you need USDC and RPDX and you need to bond and do a bunch of steps")
    perform_message("Leveraged looping on Aave")
    perform_message("pull my univ3 LP at price X and redeploy to range Y")
    perform_message("copy trade address X if token Y is greater than Z mcap with relative position size if it exceeds a certain threshold")
    perform_message("Borrow against my NFT on [Protocol X] at 60% LTV for 30 days, and stake the ETH on [Protocol Y]. 1 day before the loan is due, unstake and repay the loan")
    perform_message("compose transaction that claims rewards from existing positions (where rewards are >$2), convert and/or stake reward to appropriate vault, and update bribes based on current allocation weight")
    perform_message("sell all my CRV for Pendle on Arbitrum, lock it for 1 month, and when possible vote for the rETH pool")
    perform_message("What tokens are releasing on this chain")
    perform_message("when pool2 unlock threshold gets hit I want to remove my funds")
    perform_message("Uniswap LP management")
    perform_message("GLP farming. Like creating an algo to exit and enter at correct times")
    perform_message("Wants to loop with pendle to earn yield; have to go pendle to buy PT, then go to a market to borrow PT, then go back to pendle to buy PT; i want my LTV at 60%” - loop until it gets there")
    perform_message("Lending rate and borrowing rates for stable coins across different markets and chains are all different - thinks he would use this product to verify and quickly bridge, deposit, borrow, etc. to arb")
    perform_message("Update the range on my Uniswap position to the recommended values")
    perform_message("Monitoring uni v3 lp position and constantly balance it to keep it in a certain range or adjusting the range")
    perform_message("arbitrage bot: buy btc on x and sell on y until price equilizes")
    perform_message("If the pools total volume decreases by x% pull all of my funds in a frontrun")
    perform_message("help me snipe token A on block zero, if gas fee is above 30$ don't take the trade")
    

    # notes
    # persistent conditional actions (not time related)
    # triggers necessary for “mint”/”snipe” action
        # Contract launch trigger
        # Event or event data trigger (approximating state change unlocking functionality)
        # Function call data trigger (approximating state change unlocking functionality)
        # Transaction or transaction data trigger (approximating state change unlocking functionality)
    # "slippage" parameter for swap
    # "loop" action
    # "arbitrage" action
    # "market cap" trigger
    # “wrap” action
    # “compound” action
    # options actions and protocols
    # "balance" condition
    # "APY/yield" condition
    # multi condition triggers (AND, OR, etc)
    # ranged deposit / edit deposit range / range trigger
    # other arbitrary triggers (expected apy of position, holder distribution of token, TVL of pool, price movement %, etc)
    # funding rate condition
    # "rebalance" action
    # "copy trade" action


# notes
# more directed versions of intents, document examples
# retry on hanging / slowness from openai
# flag if context length overflows (tiktoken is a token counter mentioned on openai forum)
# potential: remove system message from beginning and persist it as latest message
# chain of thought example https://community.openai.com/t/building-hallucination-resistant-prompts/131036/6?page=2 to prevent hallucination
# unclear when to use a single mentioned protocol across multiple actions or not
# multiple token input, multiple token output (swaps, deposits)
# unclear to which other actions does a schedule/condition apply and whether those actions came before or come after
# when to assume chainName is Ethereum and when not to?

# [BACKEND]
# all interface updates
# if token needed but not there or outputToken, use from previous call. if first, assume 'all'
# if amount needed but not there or outputAmount, use from previous call. if first, assume 'all'
# if amount is token, use from previous call (perform_message("Claim Redacted rewards and relock BTRFLY"))
# if subject is 'price', assume inputToken, even though sometimes outputToken
# infer protocol if protocol field has token name (perform_message("Unstake all my plsDPX and sell it for DPX if the price if plsDPX/DPX < 0.95"))
# infer token name if poolName field has token name
# inter protocols if protocol field has chain name (perform_message("Harvest all my rewards on Arbitrum and buy ETH"), perform_message("Stake my ARB on Arbitrum"), perform_message("Harvest all my positions on Arbitrum"))
# recognize pair price as condition subject
# assume multiple conditions and times in the same query list are AND, not OR
# various formats of inputs for times
# user input autocorrect (ex. Camelor vs Camelot)
# condition or time might be within action itself
# get source chain of subsequent bridge from previous call
# hallucinates Uniswap into general swaps
# hallucinates chain names
# if first or last condition/time, assume on all, otherwise assume on below until next condition/time 
# [TRAINING] 
# doesnt recognize 'half' as an inputAmount (perform_message("Bridge 1 ETH to Base, swap half to USDC, deposit into Kyber eth-usdc pool"))



suite(17)