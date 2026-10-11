#!/bin/bash
set -uo pipefail
cd "$WASP_ROOT/repo"
"$PY" - <<'PYEOF'
from doctr.models import ocr_predictor
m = ocr_predictor(det_arch="db_resnet50", reco_arch="parseq", pretrained=False,
                  assume_straight_pages=True)
dp = m.det_predictor.pre_processor
rp = m.reco_predictor.pre_processor
print("detector  resize ve:", getattr(dp.resize, "output_size", None) or dp.resize)
print("recognizer resize ve:", getattr(rp.resize, "output_size", None) or rp.resize)
PYEOF
