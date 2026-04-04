#!/usr/bin/env python
"""Run a Bridgeport system construction job from an input JSON file."""

import argparse
import os
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description='Run a Bridgeport system construction job.'
    )
    parser.add_argument('input_json', type=str,
                        help='Path to Bridgeport input JSON configuration file.')
    args = parser.parse_args()

    if not os.path.exists(args.input_json):
        raise FileNotFoundError(
            f"Cannot find the input JSON file at: {args.input_json}"
        )

    from Bridgeport.Bridgeport import Bridgeport

    start_time = datetime.now()
    bp = Bridgeport(input_json=args.input_json)
    bp.run()
    end_time = datetime.now()
    print(f"Time to run: {end_time - start_time}")


if __name__ == '__main__':
    main()
