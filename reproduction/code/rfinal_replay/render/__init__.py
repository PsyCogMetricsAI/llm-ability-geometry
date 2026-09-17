"""C17 result renderers (C10a prepare stage).

Seven result generators for the frozen seven shell outputs:
P2-G-FIG-MTMM, P2-B-TAB-CONVERGENCE, P2-C-TAB-SCIENCE, P2-C-TAB-LOFO,
P2-F-TAB-PROBES, P2-F-FIG-PROFILES, P2-G-TAB-SUPPLEMENTS.

Design rules:
* every emitted numeric value is extracted from a declared source file via an
  explicit JSON pointer / CSV column, recorded with file sha256 in plotdata;
* no old final values and no historical images are copied or hardcoded;
* missing required inputs or a bad contract cause an explicit refusal;
* the package never writes outside its declared --outdir.
"""

SCHEMA_CONTRACT = "c17-render-contract-v1"
SCHEMA_PLOTDATA = "c17-plotdata-v1"
SCHEMA_INVENTORY = "c17-render-inventory-v1"

CANONICAL_IDS = (
    "P2-G-FIG-MTMM",
    "P2-B-TAB-CONVERGENCE",
    "P2-C-TAB-SCIENCE",
    "P2-C-TAB-LOFO",
    "P2-F-TAB-PROBES",
    "P2-F-FIG-PROFILES",
    "P2-G-TAB-SUPPLEMENTS",
)

EXIT_BAD_SCHEMA = 4
EXIT_MISSING_INPUT = 3
EXIT_RENDER_FAILURE = 5
