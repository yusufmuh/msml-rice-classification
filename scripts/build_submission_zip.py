"""Build the final Dicoding submission ZIP per PRD §11.

Structure produced (paths inside the zip):
SMSML_<NAMA_SISWA>/
├─ Eksperimen_SML_<NAMA_SISWA>.txt
├─ Membangun_model/
│    ├─ modelling.py
│    ├─ modelling_tuning.py
│    ├─ rice_preprocessing/  (train/test csvs, scaler, encoder, eda figures)
│    ├─ confusion_matrix.png
│    ├─ feature_importance.png
│    ├─ screenshoot_dashboard.jpg
│    ├─ screenshoot_artifak.jpg
│    ├─ requirements.txt
│    └─ DagsHub.txt
├─ Workflow-CI.txt
└─ Monitoring dan Logging/
      ├─ 1.bukti_serving/ (serving_inference_proof.jpg)
      ├─ 2.prometheus.yml
      ├─ 3.prometheus_exporter.py
      ├─ 4.bukti monitoring Prometheus/ (prometheus_metrics_fullpage.jpg + _metrics.html)
      ├─ 5.bukti monitoring Grafana/ (grafana_login.jpg, *_dashboard*.jpg, etc.)
      ├─ 6.bukti alerting Grafana/ (grafana_alerts.jpg)
      └─ 7.Inference.py
"""
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
NAMA = os.getenv("NAMA_SISWA", "Muhammad-Yusuf")
DEST = ROOT.parent / f"SMSML_{NAMA}"
DEST.mkdir(parents=True, exist_ok=True)
ZIP_BASE = ROOT.parent / f"SMSML_{NAMA}"

# Helper
def copy(src_rel, dst_rel):
    src = ROOT / src_rel
    dst = DEST / dst_rel
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=ignore_callback)
    elif src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        shutil.copy2(src, dst)
    else:
        print(f"  skip (missing): {src_rel}")


def ignore_callback(_, names):
    # skip non-essential large/unrelated files
    SKIP = {".ipynb_checkpoints", "__pycache__", ".DS_Store", ".git"}
    return {n for n in names if n in SKIP or n.endswith(".pyc")}


# Root-level files
print("[1/3] root files")
copy("Eksperimen_SML_Muhammad-Yusuf.txt", "Eksperimen_SML_Muhammad-Yusuf.txt")
copy("Workflow-CI.txt", "Workflow-CI.txt")

# Membangun_model (only deliverables, not the heavy raw.csv/joblib)
print("[2/3] Membangun_model")
mb = DEST / "Membangun_model"
mb.mkdir(parents=True, exist_ok=True)
copy("Membangun_model/modelling.py", "Membangun_model/modelling.py")
copy("Membangun_model/modelling_tuning.py", "Membangun_model/modelling_tuning.py")
copy("Membangun_model/confusion_matrix.png", "Membangun_model/confusion_matrix.png")
copy("Membangun_model/feature_importance.png", "Membangun_model/feature_importance.png")
copy("Membangun_model/screenshoot_dashboard.jpg", "Membangun_model/screenshoot_dashboard.jpg")
copy("Membangun_model/screenshoot_artifak.jpg", "Membangun_model/screenshoot_artifak.jpg")
copy("Membangun_model/requirements.txt", "Membangun_model/requirements.txt")
copy("Membangun_model/DagsHub.txt", "Membangun_model/DagsHub.txt")
copy("Membangun_model/rice_preprocessing", "Membangun_model/rice_preprocessing")

# Monitoring dan Logging
print("[3/3] Monitoring dan Logging")
mon = DEST / "Monitoring dan Logging"
mon.mkdir(parents=True, exist_ok=True)
copy("Monitoring dan Logging/1.bukti_serving", "Monitoring dan Logging/1.bukti_serving")
copy("Monitoring dan Logging/2.prometheus.yml", "Monitoring dan Logging/2.prometheus.yml")
copy("Monitoring dan Logging/3.prometheus_exporter.py", "Monitoring dan Logging/3.prometheus_exporter.py")
copy("Monitoring dan Logging/4.bukti monitoring Prometheus",
      "Monitoring dan Logging/4.bukti monitoring Prometheus")
copy("Monitoring dan Logging/5.bukti monitoring Grafana",
      "Monitoring dan Logging/5.bukti monitoring Grafana")
copy("Monitoring dan Logging/6.bukti alerting Grafana",
      "Monitoring dan Logging/6.bukti alerting Grafana")
copy("Monitoring dan Logging/7.Inference.py", "Monitoring dan Logging/7.Inference.py")

# zip
zip_path = ZIP_BASE.with_suffix(".zip")
if zip_path.exists():
    zip_path.unlink()
shutil.make_archive(str(ZIP_BASE), "zip", str(ZIP_BASE.parent), str(DEST.name))
print(f"\nDONE: {zip_path}")
print(f"     ({sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file()) / 1024:.1f} KB)")
