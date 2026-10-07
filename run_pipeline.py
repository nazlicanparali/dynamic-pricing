"""Run the whole project (or one step) from the command line.

    python run_pipeline.py                 # everything
    python run_pipeline.py --step forecast # just one step
"""
import argparse
import logging
import time

from pricing import ab_testing, elasticity, forecasting, optimization, rl_agent
from pricing.config import load_config

STEPS = {
    "forecast": forecasting.run,
    "elasticity": elasticity.run,
    "optimize": optimization.run,     # needs elasticity
    "rl": rl_agent.run,               # needs elasticity
    "abtest": ab_testing.run,         # needs elasticity
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["all", *STEPS], default="all")
    parser.add_argument("--config", default=None, help="path to a config file (default: config.yaml)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-22s %(message)s",
                        datefmt="%H:%M:%S")
    cfg = load_config(args.config) if args.config else load_config()

    steps = STEPS if args.step == "all" else {args.step: STEPS[args.step]}
    for name, fn in steps.items():
        t0 = time.time()
        logging.info("=== %s ===", name)
        fn(cfg)
        logging.info("%s done in %.0fs", name, time.time() - t0)


if __name__ == "__main__":
    main()
