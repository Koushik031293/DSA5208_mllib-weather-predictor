#!/usr/bin/env bash
# ============================================================
# make_deps_auto.sh — Build deps.zip for Dataproc Serverless (Python 3.11)
# Tries multiple xgboost variants/versions and picks the first that works.
# Project/Bucket: dsa5208-mllib-weather-predict
# ============================================================

set -euo pipefail

PROJECT_ID="dsa5208-mllib-weather-predict"
BUCKET="gs://dsa5208-mllib-weather-predict"
REGION="asia-southeast1"
PYVER=311   # Dataproc Serverless 2.3 runtime uses Python 3.11

BASE_REQ=$(cat <<'EOF'
numpy==1.26.4
pandas==2.2.2
pyarrow==16.1.0
scikit-learn==1.5.2
joblib==1.3.2
threadpoolctl==3.6.0
python-dateutil==2.9.0.post0
packaging==25.0
tqdm==4.66.4
# matplotlib==3.9.2   # uncomment if your job actually creates plots on the driver
EOF
)

CANDIDATES=(
  "xgboost==2.1.1"
  "xgboost-cpu==2.1.1"
  "xgboost==2.0.3"
  "xgboost-cpu==2.0.3"
  "xgboost==1.7.6"
  "xgboost-cpu==1.7.6"
  "xgboost==2.1.4"
)

clean() {
  rm -rf wheels deps deps.zip requirements.serverless.txt || true
  mkdir -p wheels deps
}

write_req() {
  local xgb="$1"
  printf "%s\n%s\n" "$BASE_REQ" "$xgb" > requirements.serverless.txt
  echo "Using requirements:"
  cat requirements.serverless.txt
}

build_with_docker() {
  echo "🛠️  Docker detected — building inside python:3.11-slim..."
  docker run --rm -v "$PWD":/work -w /work python:3.11-slim bash -lc '
    set -euo pipefail
    python -m pip install --upgrade pip
    pip install --no-cache-dir -r requirements.serverless.txt -t deps
    (cd deps && zip -rq ../deps.zip .)
  '
}

build_locally_manylinux_download() {
  echo "🐍 Building locally by downloading manylinux cp${PYVER} wheels and unzipping..."
  python3 -m pip download \
    --only-binary=:all: \
    --platform manylinux_2_17_x86_64 \
    --implementation cp \
    --python-version ${PYVER} \
    -r requirements.serverless.txt \
    -d wheels
  for f in wheels/*.whl; do unzip -q "$f" -d deps; done
  (cd deps && zip -rq ../deps.zip .)
}

upload_zip() {
  echo "☁️  Uploading deps.zip to $BUCKET/deps/deps.zip ..."
  gsutil cp deps.zip "$BUCKET/deps/deps.zip"
  echo "🔍 Verifying upload..."
  gsutil ls -lh "$BUCKET/deps/deps.zip"
}

# ----------------- MAIN -----------------
echo "📦 Building deps.zip for Dataproc Serverless (Python ${PYVER})"
echo "Project: $PROJECT_ID | Bucket: $BUCKET | Region: $REGION"
echo "=========================================================="

for xgb in "${CANDIDATES[@]}"; do
  echo "----------------------------------------------------------"
  echo "🔎 Trying candidate: $xgb"
  clean
  write_req "$xgb"

  if command -v docker >/dev/null 2>&1; then
    if build_with_docker; then
      echo "✅ Success with $xgb (Docker build)."
      upload_zip
      echo "🎉 deps.zip ready at $BUCKET/deps/deps.zip"
      exit 0
    fi
  else
    echo "🐳 Docker not found — falling back to local manylinux wheel download."
    if build_locally_manylinux_download; then
      echo "✅ Success with $xgb (local manylinux build)."
      upload_zip
      echo "🎉 deps.zip ready at $BUCKET/deps/deps.zip"
      exit 0
    fi
  fi

  echo "❌ Build failed for $xgb — trying next candidate..."
done

echo "🚨 All xgboost candidates failed. Consider using a custom container image."
exit 1