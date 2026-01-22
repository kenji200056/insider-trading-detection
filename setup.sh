#!/bin/bash
#
# インサイダー取引検出システム - セットアップスクリプト
#
# 使い方:
#   chmod +x setup.sh
#   ./setup.sh
#

set -e

echo "=================================================="
echo "  インサイダー取引検出システム セットアップ"
echo "  Insider Trading Detection System Setup"
echo "=================================================="
echo ""

# カラー定義
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 現在のディレクトリを取得
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Python確認
echo -e "${YELLOW}[1/4] Pythonの確認...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓ $PYTHON_VERSION${NC}"
else
    echo -e "${RED}✗ Python3が見つかりません。インストールしてください。${NC}"
    exit 1
fi

# 仮想環境の作成
echo ""
echo -e "${YELLOW}[2/4] 仮想環境の作成...${NC}"
if [ -d ".venv" ]; then
    echo -e "${GREEN}✓ 仮想環境は既に存在します (.venv/)${NC}"
else
    python3 -m venv .venv
    echo -e "${GREEN}✓ 仮想環境を作成しました (.venv/)${NC}"
fi

# 仮想環境の有効化
echo ""
echo -e "${YELLOW}[3/4] 依存パッケージのインストール...${NC}"
source .venv/bin/activate

# pipのアップグレード
pip install --upgrade pip -q

# 依存パッケージのインストール
pip install -r requirements.txt -q

echo -e "${GREEN}✓ 依存パッケージをインストールしました${NC}"

# MeCabの確認（macOS）
echo ""
echo -e "${YELLOW}[4/4] MeCabの確認...${NC}"
if command -v mecab &> /dev/null; then
    echo -e "${GREEN}✓ MeCabがインストールされています${NC}"
else
    echo -e "${YELLOW}⚠ MeCabが見つかりません${NC}"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  macOSの場合: brew install mecab mecab-ipadic"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "  Ubuntuの場合: sudo apt-get install mecab libmecab-dev mecab-ipadic-utf8"
    fi
fi

echo ""
echo "=================================================="
echo -e "${GREEN}  セットアップ完了!${NC}"
echo "=================================================="
echo ""
echo "使い方:"
echo ""
echo "  1. 仮想環境の有効化:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. 異常検知の実行:"
echo "     cd src/detection"
echo "     python main_detection.py"
echo ""
echo "  3. 結果の確認:"
echo "     src/detection/visualizations/"
echo ""
echo "=================================================="
