"""Every campaign builder must emit run_group_%04d.sh, the width launch.submit_script invokes.

BUGS/2026-08-24 fixed this in build_flow_md_campaign.py only. On 2026-08-30 the same defect was
still present in FOUR other builders, and DHH-v3 Step 3 died in 14 s with exit 127:
    bash: .../run_group_0000.sh: No such file or directory
A one-line format mismatch between an emitter and the shared launcher is invisible until submit
time and produces no output, so it is pinned here rather than left to the next campaign.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EMITTERS = [
    "sampling/build_harvest_campaign.py",
    "sampling/build_bond_stretch_campaign.py",
    "sampling/cases.py",
    "sampling/build_meanforce_campaign.py",
    "sampling/build_flow_md_campaign.py",
]


def test_launcher_uses_four_digits():
    src = (ROOT / "sampling/launch.py").read_text()
    assert "run_group_$(printf '%04d'" in src, "launch.py changed its group-script width"


@pytest.mark.parametrize("rel", EMITTERS)
def test_emitter_width_matches_launcher(rel):
    src = (ROOT / rel).read_text()
    # Two emission styles are in use: an f-string width (most builders) and a hardcoded
    # literal name (build_flow_md_campaign, which always has exactly one group).
    widths = re.findall(r"run_group_\{[^}]*:0(\d)d\}", src)
    literals = re.findall(r"run_group_(\d+)\.sh", src)
    assert widths or literals, (
        f"{rel} emits no run_group_* script; update this test if that is intentional")
    assert all(w == "4" for w in widths), (
        f"{rel} emits run_group_%0{(widths or ['?'])[0]}d.sh but "
        f"launch.submit_script invokes %04d")
    assert all(len(l) == 4 for l in literals), (
        f"{rel} hardcodes run_group_{literals} but launch.submit_script invokes %04d")
