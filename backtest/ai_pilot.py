"""Phase 13: can an AI's judgment pick the winners among screened candidates? A hindsight-free pilot.

Rules were written down and committed before the model was downloaded or run.

Model: Meta Llama 3.1 8B Instruct (4-bit GGUF), whose training data ends in December 2023 (Meta model card),
run locally with llama.cpp, temperature 0. It cannot know what happened to any stock after that.

Candidates: every entry dated 2024-02-01 or later from
  (a) tradeable big-drop insider clusters (>= 3 insiders, >= $200K, stock down >= 50% in 6 months,
      >= $100K a day traded), and
  (b) S&P 500 officer/director purchases >= $100K (test L1), and
  (c) the DFV screen without the smart-money condition (test D4).
Input: a fact sheet built only from data public at the entry date (company name, sector code, market cap,
6- and 12-month price change, valuation, cash, debt, trailing earnings, revenue trend, short interest,
insider purchases with titles and amounts, recent 8-K item codes). Same prompt for every candidate.
Output: JSON with a 1-10 score, buy / pass, and three short reasons.

Measures (pre-registered): Spearman correlation between score and the 6-month return minus SPY (entries
with a complete 6 months); average 6-month return minus SPY for score >= 7 against score <= 4; and
"buy" against "pass". Small sample: treated as a pilot, not proof.
"""
