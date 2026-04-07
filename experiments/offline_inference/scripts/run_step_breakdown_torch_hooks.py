#!/usr/bin/env python3

from run_step_breakdown_microbench import main


if __name__ == "__main__":
    main(include_baseline_arg=False, default_baseline="torch_hooks", default_baseline_label="torch_hooks")
