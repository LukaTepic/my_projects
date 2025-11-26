

import os
import json
import logging
import sys
import time
import struct
from typing import Tuple, Optional, List
from decimal import Decimal, getcontext

from web3 import Web3
from web3.exceptions import ContractLogicError
import requests

getcontext().prec = 50 

try:
    from scipy.stats import beta
except ImportError:
    print("FATAL ERROR: Scipy library not found. Please install it using 'pip install scipy'")
    sys.exit(1)


class JsonFileDb:
    def __init__(self, filepath='db.json'):
        self._filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)
        self._data = self._load()

    def _load(self):
        try:
            with open(self._filepath, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self):
        with open(self._filepath, 'w') as f:
            json.dump(self._data, f, indent=2)

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __setitem__(self, key, value):
        self._data[key] = value
        self._save()

    def __getitem__(self, key):
        return self._data[key]

db = JsonFileDb()


def setup_logging(wallet_id=None):
    log_format = f'%(asctime)s - %(levelname)s - [{wallet_id or "global"}] - %(message)s'
    logger = logging.getLogger(wallet_id or "VaultStrategyManager")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(log_format)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        os.makedirs("logs", exist_ok=True)
        log_file = f"logs/strategy_{wallet_id or 'main'}.log"
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

class VaultStrategyManager:

    def __init__(self):
        self.WALLET_ID = os.getenv('id')
        if not self.WALLET_ID:
            logging.critical("FATAL: Environment variable 'id' is not set. Cannot operate.")
            sys.exit(1)
        
        self.logger = setup_logging(self.WALLET_ID)

        self.KEY_STORE = os.getenv('KEY_STORE', 'default_strategy')
        self.RPC_URL = os.getenv('RPC_URL')
        self.PRIVATE_KEY = os.getenv('PRIVATE_KEY')
        self.VAULT_CONTRACT_ADDRESS = os.getenv('VAULT_CONTRACT_ADDRESS')
        self.LBP_CONTRACT_ADDRESS = os.getenv('LBP_CONTRACT_ADDRESS')

        self.DEPLOY_PERCENTAGE_X = float(os.getenv('DEPLOY_PERCENTAGE_X', '10.0'))
        self.DEPLOY_PERCENTAGE_Y = float(os.getenv('DEPLOY_PERCENTAGE_Y', '10.0'))
        
        self.MIN_BALANCE_PERCENTAGE_X = float(os.getenv('MIN_BALANCE_PERCENTAGE_X', '5.0'))
        self.MIN_BALANCE_PERCENTAGE_Y = float(os.getenv('MIN_BALANCE_PERCENTAGE_Y', '5.0'))

        self.UP_BINS = int(os.getenv('UP_BINS', 10))
        self.DOWN_BINS = int(os.getenv('DOWN_BINS', 10))
        self.SLIPPAGE_ACTIVE_ID = int(os.getenv('SLIPPAGE_ACTIVE_ID', 5))

        self.BETA_ALPHA = float(os.getenv('BETA_ALPHA', 0.6))
        self.BETA_BETA = float(os.getenv('BETA_BETA', 0.6))

        self.RANGE_LOW_KEY = f"{self.KEY_STORE}_range_low"
        self.RANGE_HIGH_KEY = f"{self.KEY_STORE}_range_high"
        self.DEPLOYMENT_TYPE_KEY = f"{self.KEY_STORE}_deployment_type"

        self.w3 = Web3(Web3.HTTPProvider(self.RPC_URL))
        self.account = self.w3.eth.account.from_key(self.PRIVATE_KEY)
        self.logger.info(f"Manager for '{self.WALLET_ID}' using wallet: {self.account.address}")
        self.logger.info(f"Deployment Strategy: Target {self.DEPLOY_PERCENTAGE_X}% of total value as Token X.")
        self.logger.info(f"Deployment Strategy: Target {self.DEPLOY_PERCENTAGE_Y}% of total value as Token Y.")
        self.logger.info(f"Safeguard: Maintain at least {self.MIN_BALANCE_PERCENTAGE_X}% for Token X and {self.MIN_BALANCE_PERCENTAGE_Y}% for Token Y.")

        self.erc20_abi = self._load_abi_from_file("abi/erc20_contract_abi.json")
        self.lbp_abi = self._load_abi_from_file("abi/lbpair_contract_abi.json")
        self.vault_abi = self._load_abi_from_file("abi/strategy_contract_abi.json")

        if not all([self.LBP_CONTRACT_ADDRESS, self.VAULT_CONTRACT_ADDRESS]):
            raise ValueError("VAULT_CONTRACT_ADDRESS and LBP_CONTRACT_ADDRESS must be set.")

        self.pair_contract = self.w3.eth.contract(address=Web3.to_checksum_address(self.LBP_CONTRACT_ADDRESS), abi=self.lbp_abi)
        self.strategy_contract = self.w3.eth.contract(address=Web3.to_checksum_address(self.VAULT_CONTRACT_ADDRESS), abi=self.vault_abi)
        self.logger.info("Contracts initialized successfully.")

        self.error_selectors = self._load_error_selectors()

        self.logger.info("Fetching token addresses from contracts...")
        self.token_x_address = self.strategy_contract.functions.getTokenX().call()
        self.token_y_address = self.strategy_contract.functions.getTokenY().call()

        self.token_x_contract = self.w3.eth.contract(address=self.token_x_address, abi=self.erc20_abi)
        self.token_y_contract = self.w3.eth.contract(address=self.token_y_address, abi=self.erc20_abi)
        self.decimals_x = self.token_x_contract.functions.decimals().call()
        self.decimals_y = self.token_y_contract.functions.decimals().call()
        self.logger.info(f"Token X: {self.token_x_address} ({self.decimals_x} decimals)")
        self.logger.info(f"Token Y: {self.token_y_address} ({self.decimals_y} decimals)")

    def _check_for_manual_trigger(self) -> bool:

        script_dir = os.path.dirname(os.path.abspath(__file__))
        trigger_file = os.path.join(script_dir, "triggers", f"{self.WALLET_ID}_rebalance.trigger")
        
        if os.path.exists(trigger_file):
            self.logger.warning(f"MANUAL TRIGGER DETECTED at {trigger_file}!")
            try:
                os.remove(trigger_file)
                self.logger.info("Successfully removed trigger file.")
            except OSError as e:
                self.logger.error(f"Error removing trigger file: {e}")
            return True
        return False

    def manage_liquidity(self):
        try:
            manual_trigger = self._check_for_manual_trigger()
            
            active_id = self.pair_contract.functions.getActiveId().call()
            stored_low = db.get(self.RANGE_LOW_KEY, 0)
            stored_high = db.get(self.RANGE_HIGH_KEY, 0)
            
            last_deployment_type = db.get(self.DEPLOYMENT_TYPE_KEY, "symmetrical")
            self.logger.info(f"Current Active Bin: {active_id}. Stored Position: [{stored_low}, {stored_high}]. Last deploy: {last_deployment_type}")

            buffer_low, buffer_high = stored_low, stored_high
            if last_deployment_type == 'single_x':
                buffer_low -= 1 # Add a buffer for single-sided X deposits
            elif last_deployment_type == 'single_y':
                buffer_high += 1 # Add a buffer for single-sided Y deposits

            is_out_of_range = not (buffer_low <= active_id <= buffer_high and stored_high != 0)

            if not manual_trigger and not is_out_of_range:
                self.logger.info(f"Position is active within buffered range [{buffer_low}, {buffer_high}]. No action needed.")
                return

            if manual_trigger:
                self.logger.warning("Rebalancing due to manual trigger.")
            elif is_out_of_range:
                self.logger.warning("Position is out of buffered range. Triggering rebalance.")

            price_x_in_y = self.get_onchain_price()
            if price_x_in_y is None or price_x_in_y == 0:
                self.logger.error("Could not get a valid on-chain price. Aborting rebalance.")
                return

            price_y_in_usd = 1.0
            price_x_in_usd = price_x_in_y * price_y_in_usd

            x_balance, y_balance = self.strategy_contract.functions.getBalances().call()
            x_balance_usd = (x_balance / 10**self.decimals_x) * price_x_in_usd
            y_balance_usd = (y_balance / 10**self.decimals_y) * price_y_in_usd
            total_balance_usd = x_balance_usd + y_balance_usd
            self.logger.info(f"Total idle vault value: ${total_balance_usd:.2f} (${x_balance_usd:.2f} of X, ${y_balance_usd:.2f} of Y)")

            if total_balance_usd == 0:
                self.logger.warning("Total balance is zero. Cannot deploy.")
                return
            
            target_deploy_usd_x = total_balance_usd * (self.DEPLOY_PERCENTAGE_X / 100.0)
            target_deploy_usd_y = total_balance_usd * (self.DEPLOY_PERCENTAGE_Y / 100.0)
            self.logger.info(f"Deployment Targets: ${target_deploy_usd_x:.2f} of Token X, ${target_deploy_usd_y:.2f} of Token Y.")


            min_balance_usd_x = total_balance_usd * (self.MIN_BALANCE_PERCENTAGE_X / 100.0)
            min_balance_usd_y = total_balance_usd * (self.MIN_BALANCE_PERCENTAGE_Y / 100.0)
            self.logger.info(f"Safeguard Thresholds: Min idle X: ${min_balance_usd_x:.2f}, Min idle Y: ${min_balance_usd_y:.2f}.")
            
            x_check_passed = (x_balance_usd - target_deploy_usd_x) >= min_balance_usd_x
            y_check_passed = (y_balance_usd - target_deploy_usd_y) >= min_balance_usd_y

            master_lower = active_id - self.DOWN_BINS
            master_upper = active_id + self.UP_BINS
            self.logger.info(f"Generating master distribution for full range [{master_lower}, {master_upper}]")
            master_weights_x, master_weights_y = self._generate_beta_distribution_arrays(master_lower, master_upper, active_id)

            value_x_to_send_usd, value_y_to_send_usd = 0.0, 0.0
            final_lower, final_upper = 0, 0
            unnormalized_weights_x, unnormalized_weights_y = [], []
            deployment_type = "symmetrical"

            if x_check_passed and y_check_passed:
                self.logger.info("Both tokens passed safeguard. Deploying symmetrically.")
                deployment_type, value_x_to_send_usd, value_y_to_send_usd = "symmetrical", target_deploy_usd_x, target_deploy_usd_y
                final_lower, final_upper = master_lower, master_upper
                unnormalized_weights_x, unnormalized_weights_y = master_weights_x, master_weights_y
            elif y_check_passed:
                self.logger.warning("Token X under safeguard. Deploying ONLY Token Y.")
                deployment_type, value_y_to_send_usd = "single_y", target_deploy_usd_y
                final_lower, final_upper = master_lower, active_id
                start_idx, end_idx = final_lower - master_lower, final_upper - master_lower + 10
                unnormalized_weights_x, unnormalized_weights_y = master_weights_x[start_idx:end_idx], master_weights_y[start_idx:end_idx]
            elif x_check_passed:
                self.logger.warning("Token Y under safeguard. Deploying ONLY Token X.")
                deployment_type, value_x_to_send_usd = "single_x", target_deploy_usd_x
                final_lower, final_upper = active_id, master_upper
                start_idx, end_idx = final_lower - master_lower, final_upper - master_lower + 10
                unnormalized_weights_x, unnormalized_weights_y = master_weights_x[start_idx:end_idx], master_weights_y[start_idx:end_idx]
            else:
                self.logger.error("Both tokens are under safeguard limits. Aborting rebalance cycle.")
                return
            
            self.logger.info(f"Final plan: deploy ${value_x_to_send_usd:.4f} X, ${value_y_to_send_usd:.4f} Y across [{final_lower}, {final_upper}].")

            final_dist_x = self._normalize_weights(unnormalized_weights_x)
            final_dist_y = self._normalize_weights(unnormalized_weights_y)
            distro_bytes = self._pack_distribution_to_bytes(final_dist_x, final_dist_y)

            x_to_send = int((Decimal(value_x_to_send_usd) / Decimal(price_x_in_usd)) * (10**self.decimals_x)) if price_x_in_usd > 0 else 0
            y_to_send = int((Decimal(value_y_to_send_usd) / Decimal(price_y_in_usd)) * (10**self.decimals_y)) if price_y_in_usd > 0 else 0
            self.logger.info(f"Final amounts: X={x_to_send / 10**self.decimals_x:.4f}, Y={y_to_send / 10**self.decimals_y:.4f}")

            self.rebalance(final_lower, final_upper, active_id, x_to_send, y_to_send, distro_bytes, deployment_type)

        except Exception as e:
            self.logger.exception(f"Unhandled error in manage_liquidity: {e}")


    def get_onchain_price(self) -> Optional[float]:
        try:
            active_id = self.pair_contract.functions.getActiveId().call()
            price_as_uint = self.pair_contract.functions.getPriceFromId(active_id).call()
            raw_ratio = Decimal(price_as_uint) / Decimal(2**128)
            price = float(raw_ratio * (Decimal(10)**(self.decimals_x - self.decimals_y)))
            self.logger.info(f"On-chain price calculated: 1 TokenX = {price:.6f} TokenY")
            return price
        except Exception as e:
            self.logger.error(f"Could not calculate on-chain price: {e}")
            return None

    def _normalize_weights(self, weights: List[float]) -> List[int]:
        if not weights: return []
        weights_dec = [Decimal(w) for w in weights]
        total_weight_dec = sum(weights_dec)
        if total_weight_dec == Decimal(0): return [0] * len(weights)
        
        TEN_E_18 = Decimal(10**18)
        dist = [int(w_dec * TEN_E_18 / total_weight_dec) for w_dec in weights_dec]
        remainder = 10**18 - sum(dist)
        
        if remainder != 0 and weights:
            max_idx = weights.index(max(weights))
            dist[max_idx] += remainder
        return dist

    def _generate_beta_distribution_arrays(self, lower_range: int, upper_range: int, active_id: int) -> Tuple[List[float], List[float]]:
        num_bins = upper_range - lower_range + 1
        weights_x, weights_y = [0.0] * num_bins, [0.0] * num_bins

        if not (lower_range <= active_id <= upper_range) or num_bins <= 0:
            self.logger.warning("Active ID out of range or invalid bin count. Using uniform distribution.")
            return ([1.0] * num_bins, [1.0] * num_bins) if num_bins > 0 else ([], [])

        bin_edges = [i / num_bins for i in range(num_bins + 1)]
        cdf_values = beta.cdf(bin_edges, self.BETA_ALPHA, self.BETA_BETA)
        shape_weights = [cdf_values[i + 1] - cdf_values[i] for i in range(num_bins)]
        
        active_bin_index = active_id - lower_range
        for i in range(num_bins):
            if i < active_bin_index:
                weights_y[i] = shape_weights[i]
            elif i == active_bin_index:
                weights_x[i] = shape_weights[i] / 2.0
                weights_y[i] = shape_weights[i] / 2.0
            else: 
                weights_x[i] = shape_weights[i]
        return weights_x, weights_y

    def _pack_distribution_to_bytes(self, dist_x: List[int], dist_y: List[int]) -> bytes:
        result = b''
        num_bins = min(len(dist_x), len(dist_y))
        for i in range(num_bins):
            result += struct.pack(">Q", dist_x[i])
            result += struct.pack(">Q", dist_y[i])
        return result

    def rebalance(self, new_lower: int, new_upper: int, active_id: int, amount_x: int, amount_y: int, distro: bytes, deployment_type: str):
        self.logger.info(f"Preparing rebalance for range [{new_lower}, {new_upper}]")
        if amount_x <= 0 and amount_y <= 0:
             self.logger.warning("Both deployment amounts are zero or negative. Nothing to do.")
             return

        try:
            function_call = self.strategy_contract.functions.rebalance(
                new_lower, new_upper, active_id, self.SLIPPAGE_ACTIVE_ID, amount_x, amount_y, distro)

            self.logger.info("Simulating rebalance transaction...")
            function_call.call({'from': self.account.address})

            self.logger.info("✅ Simulation successful! Submitting transaction...")
            estimated_gas = function_call.estimate_gas({'from': self.account.address})

            tx = function_call.build_transaction({
                'from': self.account.address,
                'gas': int(estimated_gas * 1.25),
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
            })

            signed_tx = self.w3.eth.account.sign_transaction(tx, self.PRIVATE_KEY)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)

            self.logger.info(f"Transaction sent: {tx_hash.hex()}. Waiting for receipt...")
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)

            if receipt.status == 1:
                self.logger.info(f"✅✅✅ Rebalance successful! New range: [{new_lower}, {new_upper}]")
                db[self.RANGE_LOW_KEY] = new_lower
                db[self.RANGE_HIGH_KEY] = new_upper
                db[self.DEPLOYMENT_TYPE_KEY] = deployment_type
            else:
                self.logger.error(f"Transaction failed. Receipt: {receipt}")

        except ContractLogicError as e:
            error_msg = str(e)
            error_name = f"Unknown revert reason: {error_msg}"
            try:
                data_str = error_msg.split("'data': '")[1].split("'")[0]
                selector = data_str[:10]
                error_name = self.error_selectors.get(selector, f"Unknown selector: {selector}")
            except Exception: pass
            self.logger.error(f"Rebalance simulation failed with revert: '{error_name}'")
        except Exception as e:
            self.logger.exception(f"Error during rebalance execution: {e}")

    def _load_abi_from_file(self, filename: str) -> list:
        abi_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        try:
            with open(abi_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.critical(f"FATAL: Error loading ABI {abi_path}: {e}", exc_info=True)
            sys.exit(1)

    def _load_error_selectors(self) -> dict:
        selectors = {}
        for abi in [self.vault_abi, self.lbp_abi]:
            for item in abi:
                if item.get('type') == 'error':
                    name = item.get('name')
                    inputs = ','.join([inp['type'] for inp in item.get('inputs', [])])
                    signature = f"{name}({inputs})"
                    selector = self.w3.keccak(text=signature).hex()[:10]
                    selectors[selector] = signature
        return selectors

def main():
    """Main entry point for the script."""
    try:
        manager = VaultStrategyManager()
        manager.logger.info(f"--- Starting Liquidity Manager for {manager.WALLET_ID} ---")
        while True:
            manager.manage_liquidity()
            manager.logger.info("--- Cycle Complete, Waiting 10 seconds ---")
            time.sleep(10)
    except Exception as e:
        logging.critical(f"A critical error occurred in main: {e}", exc_info=True)
        time.sleep(60)
        
if __name__ == "__main__":
    main()