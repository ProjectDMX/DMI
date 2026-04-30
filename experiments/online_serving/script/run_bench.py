import argparse
from vllm.benchmarks.serve import main, add_cli_args
parser = argparse.ArgumentParser()
add_cli_args(parser)
args = parser.parse_args()
main(args)
