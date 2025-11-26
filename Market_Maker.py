# Market_Maker.py

import os
import time
import sys
import subprocess
import threading
import signal
import logging
import json
import boto3
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("strategy_server.log")
    ]
)
logger = logging.getLogger("StrategyServer")

def get_ssm_parameter(name):
    """Fetches a parameter from AWS Systems Manager Parameter Store."""
    try:
        ssm_client = boto3.client('ssm', region_name='eu-north-1')
        response = ssm_client.get_parameter(Name=name, WithDecryption=True)
        return response['Parameter']['Value']
    except Exception as e:
        logger.error(f"FATAL: Could not fetch SSM parameter '{name}'. Check IAM role permissions. Error: {e}")
        sys.exit(1)

logger.info("Fetching configuration from AWS Parameter Store...")
RPC_URL = get_ssm_parameter('/market-maker/rpc-url')
CHECK_INTERVAL = 60
MAX_RESTART_ATTEMPTS = 5


wallet_configs = [
    {
        'id': 'Wallet_1_Volatile',
        'PRIVATE_KEY': get_ssm_parameter('/market-maker/private-key-4'),
        'RPC_URL': RPC_URL,
        'VAULT_CONTRACT_ADDRESS': get_ssm_parameter('/market-maker/vault-contract'),
        'LBP_CONTRACT_ADDRESS': get_ssm_parameter('/market-maker/lbp-contract'),
        'KEY_STORE': 'MyVolatileStrategy',

        'DEPLOY_PERCENTAGE_X': '1',  # Deploy 2.5% of total vault value as Token X
        'DEPLOY_PERCENTAGE_Y': '1',   # Deploy 5% of total vault value as Token Y  
        
        'MIN_BALANCE_PERCENTAGE_X': '1', # Keep at least X% of total value as a buffer for Token X
        'MIN_BALANCE_PERCENTAGE_Y': '1', # Keep at least Y% of total value as a buffer for Token Y

        'UP_BINS': '25',
        'DOWN_BINS': '25',
        'BETA_ALPHA': '0.7',
        'BETA_BETA': '0.7',
        'SLIPPAGE_ACTIVE_ID': '3'
    }
]

logger.info("Configuration fetched successfully.")

def trigger_rebalance(wallet_id=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    triggers_dir = os.path.join(script_dir, "triggers")
    os.makedirs(triggers_dir, exist_ok=True)

    wallets_to_trigger = []
    if wallet_id:
        if any(c['id'] == wallet_id for c in wallet_configs):
            wallets_to_trigger.append(wallet_id)
        else:
            return f"Error: Wallet ID '{wallet_id}' not found in configuration."
    else:
        wallets_to_trigger = [config['id'] for config in wallet_configs]

    for w_id in wallets_to_trigger:
        trigger_file = os.path.join(triggers_dir, f"{w_id}_rebalance.trigger")
        logger.info(f"Creating trigger file for wallet {w_id} at: {trigger_file}")
        with open(trigger_file, 'w') as f:
            f.write(str(time.time()))
    return f"Rebalance triggered for: {', '.join(wallets_to_trigger)}"


class StrategyManagerProcess:
    def __init__(self, config):
        self.config = config
        self.process = None
        self.last_restart_time = 0
        self.restart_count = 0

    def start(self):
        try:
            current_time = time.time()
            if current_time - self.last_restart_time < 60:
                logger.warning(f"Attempted to restart {self.config['id']} too quickly. Waiting...")
                return False

            if self.restart_count >= MAX_RESTART_ATTEMPTS:
                logger.error(f"Hit max restarts for {self.config['id']}. Waiting 30 mins.")
                time.sleep(1800)
                self.restart_count = 0

            logger.info(f"Starting strategy management for {self.config['id']}")

            cmd = [sys.executable, "Maker_Strategy.py"]

            env = os.environ.copy()
            env.update({k: str(v) for k, v in self.config.items() if v is not None})

            os.makedirs("logs", exist_ok=True)
            with open(f"logs/{self.config['id']}_config.json", 'w') as f:
                safe_config = {k: v for k, v in self.config.items() if k != 'PRIVATE_KEY'}
                json.dump(safe_config, f, indent=2)

            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid
            )

            threading.Thread(target=self._log_output, args=(self.process.stdout, logging.INFO), daemon=True).start()
            threading.Thread(target=self._log_output, args=(self.process.stderr, logging.ERROR), daemon=True).start()

            self.last_restart_time = time.time()
            self.restart_count += 1
            logger.info(f"Process started for {self.config['id']} (PID: {self.process.pid})")
            return True
        except Exception as e:
            logger.error(f"Failed to start process for {self.config['id']}: {e}")
            return False

    def _log_output(self, stream, level):
        for line in iter(stream.readline, ''):
            logger.log(level, f"[{self.config['id']}] {line.strip()}")

    def stop(self):
        if self.process:
            logger.info(f"Stopping strategy management for {self.config['id']}")
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning(f"Process for {self.config['id']} did not terminate gracefully. Killing...")
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait(timeout=5)
            except Exception as e:
                logger.error(f"Error stopping process for {self.config['id']}: {e}")
            finally:
                self.process = None

    def is_running(self):
        if self.process is None:
            return False
        return self.process.poll() is None

def signal_handler(sig, frame):
    logger.info("Received shutdown signal. Stopping all processes...")
    for p in processes:
        p.stop()
    sys.exit(0)

def monitor_processes():
    for process in processes:
        if not process.is_running():
            logger.error(f"Process for {process.config['id']} stopped unexpectedly. Restarting...")
            process.stop()
            time.sleep(5)
            process.start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Strategy Server Management')
    parser.add_argument('--trigger', help='Trigger rebalance for a specific wallet ID')
    parser.add_argument('--trigger-all', action='store_true', help='Trigger rebalance for all wallets')
    args = parser.parse_args()

    if args.trigger:
        print(trigger_rebalance(args.trigger))
        sys.exit(0)
    elif args.trigger_all:
        print(trigger_rebalance())
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    processes = []
    for config in wallet_configs:
        if not config.get('PRIVATE_KEY'):
            logger.warning(f"Skipping {config['id']} due to missing PRIVATE_KEY")
            continue
        process = StrategyManagerProcess(config)
        if process.start():
            processes.append(process)
            time.sleep(5)

    try:
        while True:
            monitor_processes()
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        for process in processes:
            process.stop()