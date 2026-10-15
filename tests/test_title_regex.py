import pytest

from backtest.signals import is_top_exec, fund_exclusion_reason

MATCH = [
    "CEO", "Pres, CEO", "COB, Dir", "COB", "CFO", "Dir, CEO", "Chief Executive Officer",
    "President and CEO", "President & Chief Executive Officer", "Chairman & CEO",
    "Chairman of the Board", "Executive Chairman", "Co-Chairman", "Chairwoman", "Chair",
    "EVP and CFO", "SVP, Chief Financial Officer", "Chief Financial Officer and Treasurer",
    "Principal Financial Officer", "Co-CEO", "C.E.O.", "Interim CEO", "Dir, 10%, Chief Executive Officer",
    "Chairman, President, CEO", "Dir, 10%, Executive Chairman", "CEO & Director",
]
NO_MATCH = [
    "Dir", "10%", "Dir, 10%", "President", "COO", "Chief Operating Officer", "EVP",
    "Vice Chairman", "Vice-Chairman", "Vice Chair", "Deputy Chairman", "Chief Accounting Officer",
    "General Counsel", "VP Finance", "Former CEO", "Chairman Emeritus", "Director, Audit Committee",
    "CTO", "Chief Technology Officer", "SVP", "", None, "Principal Accounting Officer", "Secretary",
    "Chief Investment Officer",
]


@pytest.mark.parametrize("title", MATCH)
def test_top_exec_titles_match(title):
    assert is_top_exec(title)


@pytest.mark.parametrize("title", NO_MATCH)
def test_other_titles_do_not_match(title):
    assert not is_top_exec(title)


def test_fund_exclusion():
    rng = (6720, 6799)
    assert fund_exclusion_reason("ACME REIT", None, 6798, rng) == "SIC 6798"
    assert fund_exclusion_reason("Gabelli Equity Trust", None, None, rng) == "name"
    assert fund_exclusion_reason("Some Income Fund Inc", None, 1000, rng) == "name"
    assert fund_exclusion_reason("NORTHERN TRUST CORP", None, 6022, rng) is None   # bank
    assert fund_exclusion_reason("TrustCo Bank Corp NY", None, 6022, rng) is None
    assert fund_exclusion_reason("Apple Inc", None, 3571, rng) is None
