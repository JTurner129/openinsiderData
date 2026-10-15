"""Run the insider-buying backtest end to end, or a single phase.

    python run_backtest.py            # all phases
    python run_backtest.py 3 4 5      # only phases 3-5 (re-uses earlier outputs)

Phases: 1 data (SEC Form 4 purchases), 2 signals, 3 prices, 4 returns,
5 analysis, 6 realistic trading and filter tests, 7 report. Every phase caches its downloads, so re-running after an
interruption resumes instead of starting over.
"""
import sys

from backtest import analysis, filters, prices, report, returns, signals
from backtest.common import load_config
import sec_insider_data

PHASES = {
    1: ('Data', lambda cfg: sec_insider_data.main()),
    2: ('Signals', signals.build_signals),
    3: ('Prices', prices.run),
    4: ('Returns', returns.run),
    5: ('Analysis', analysis.run),
    6: ('Realistic trading and filter tests', filters.run),
    7: ('Report', report.build),
}

if __name__ == '__main__':
    wanted = [int(a) for a in sys.argv[1:]] or list(PHASES)
    cfg = load_config()
    for p in wanted:
        name, fn = PHASES[p]
        print(f'\n=== Phase {p}: {name} ===', flush=True)
        fn(cfg)
