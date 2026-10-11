"""Fresh-process execution of one independent chain or atomic diagnostic job."""
import json
import os
from pathlib import Path
import sys

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
# Bound each fresh worker before NumPy/native imports; leave RAM for the runtime.
if os.environ.get("DIDBCF_WORKER_MEMORY_BYTES"):
    import resource
    ceiling = int(os.environ["DIDBCF_WORKER_MEMORY_BYTES"])
    resource.setrlimit(resource.RLIMIT_AS, (ceiling, ceiling))
from recovery_engine import activate
activate()
from recovery_core import execute_unit

if __name__ == "__main__":
    request = json.loads(Path(sys.argv[1]).read_text())
    execute_unit(request["unit"], request["identity"], Path(request["workdir"]))
