#!/bin/bash
set -e

###############################################################################
# download_pretrained.sh
#
# NerfBaselines (HuggingFace)에서 3D Gaussian Splatting 프리트레인 모델을 받아
# 블렌더 데이터셋 디렉토리에 gs.ply로 배치하는 스크립트
#
# 사용법:
#   ./download_pretrained.sh                          # 기본 경로 사용
#   ./download_pretrained.sh /path/to/blender/dataset  # 데이터셋 경로 지정
###############################################################################

BLENDER_DIR="${1:-/data/wgsong/dataset/blender}"
SCENES=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")
BASE_URL="https://huggingface.co/nerfbaselines/nerfbaselines/resolve/main/gaussian-splatting/blender"
TMPDIR=$(mktemp -d)

trap "rm -rf $TMPDIR" EXIT

echo "=========================================="
echo " 3DGS Pretrained Model Downloader"
echo " Source: NerfBaselines (HuggingFace)"
echo " Target: ${BLENDER_DIR}/{scene}/gs.ply"
echo "=========================================="
echo ""

# Step 1: 씬 디렉토리 생성
echo "[Step 1] Creating scene directories..."
for scene in "${SCENES[@]}"; do
    mkdir -p "${BLENDER_DIR}/${scene}"
done
echo "Done."
echo ""

# Step 2: 다운로드 및 PLY 추출
echo "[Step 2] Downloading and extracting PLY files..."
echo ""

FAIL_COUNT=0

for scene in "${SCENES[@]}"; do
    ZIP_URL="${BASE_URL}/${scene}.zip"
    ZIP_FILE="${TMPDIR}/${scene}.zip"
    DST_FILE="${BLENDER_DIR}/${scene}/gs.ply"

    # 이미 존재하면 스킵
    if [ -f "$DST_FILE" ]; then
        SIZE=$(du -h "$DST_FILE" | cut -f1)
        echo "  [SKIP] ${scene} - gs.ply already exists (${SIZE})"
        continue
    fi

    echo -n "  [DOWN] ${scene} ... "
    if ! wget -q --show-progress -O "$ZIP_FILE" "$ZIP_URL" 2>&1; then
        echo "FAILED (download error)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    # Python zipfile로 point_cloud.ply 추출 (unzip 불필요)
    if ! python3 -c "
import zipfile, sys, os
with zipfile.ZipFile('${ZIP_FILE}', 'r') as z:
    ply = [n for n in z.namelist() if n.endswith('point_cloud.ply') and '30000' in n]
    if not ply:
        ply = [n for n in z.namelist() if n.endswith('point_cloud.ply')]
    if not ply:
        print('NO PLY FOUND', file=sys.stderr)
        sys.exit(1)
    with z.open(ply[0]) as src, open('${DST_FILE}', 'wb') as dst:
        dst.write(src.read())
" 2>/dev/null; then
        echo "FAILED (extraction error)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    # zip 즉시 삭제 (디스크 절약)
    rm -f "$ZIP_FILE"

    SIZE=$(du -h "$DST_FILE" | cut -f1)
    echo "  [OK]   ${scene} -> gs.ply (${SIZE})"
done

echo ""

# Step 3: 결과 검증
echo "[Step 3] Verification"
echo "-------------------------------------------"
OK_COUNT=0
for scene in "${SCENES[@]}"; do
    DST_FILE="${BLENDER_DIR}/${scene}/gs.ply"
    if [ -f "$DST_FILE" ]; then
        SIZE=$(du -h "$DST_FILE" | cut -f1)
        echo "  [OK]      ${scene}/gs.ply  (${SIZE})"
        OK_COUNT=$((OK_COUNT + 1))
    else
        echo "  [MISSING] ${scene}/gs.ply"
    fi
done
echo "-------------------------------------------"
echo "Result: ${OK_COUNT}/${#SCENES[@]} scenes ready"

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo "WARNING: ${FAIL_COUNT} scene(s) failed. Re-run the script to retry."
    exit 1
fi

echo ""
echo "Done! All PLY files placed in ${BLENDER_DIR}/{scene}/gs.ply"
