#!/bin/bash
# 修复 Qlib 缺失依赖的脚本

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== 修复 Qlib 缺失依赖 ===${NC}\n"

# 激活虚拟环境（如果存在）
if [ -d "venv" ]; then
    echo -e "${YELLOW}激活虚拟环境...${NC}"
    source venv/bin/activate
fi

# 升级 pip
echo -e "${YELLOW}升级 pip...${NC}"
pip install --upgrade pip setuptools wheel

# 安装 setuptools_scm（最关键的缺失依赖）
echo -e "\n${YELLOW}安装 setuptools_scm...${NC}"
pip install setuptools_scm

# 安装 Qlib 的其他依赖
echo -e "\n${YELLOW}安装 Qlib 的其他依赖...${NC}"
echo -e "${BLUE}这可能需要一些时间，请耐心等待...${NC}"

dependencies=(
    "cvxpy"
    "dill"
    "fire"
    "gym"
    "jupyter"
    "lightgbm"
    "matplotlib"
    "mlflow"
    "nbconvert"
    "pymongo"
    "python-redis-lock"
    "redis"
    "ruamel.yaml>=0.17.38"
)

for dep in "${dependencies[@]}"; do
    echo -e "${YELLOW}安装 $dep...${NC}"
    pip install "$dep" || echo -e "${RED}⚠️  $dep 安装失败，继续安装其他依赖...${NC}"
done

# 验证安装
echo -e "\n${GREEN}验证 Qlib 安装...${NC}"
python -c "
try:
    import qlib
    from qlib.config import REG_CN
    from qlib.data import D
    version = getattr(qlib, '__version__', 'unknown')
    print(f'✅ Qlib 安装成功！版本: {version}')
    
    # 检查关键依赖
    try:
        import cvxpy
        print('✅ cvxpy 可用')
    except ImportError:
        print('⚠️  cvxpy 不可用')
    
    try:
        import lightgbm
        print('✅ lightgbm 可用')
    except ImportError:
        print('⚠️  lightgbm 不可用')
        
except ImportError as e:
    print(f'❌ Qlib 导入失败: {e}')
    exit(1)
" || {
    echo -e "${RED}Qlib 验证失败，请检查错误信息${NC}"
    exit 1
}

echo -e "\n${GREEN}✅ Qlib 依赖修复完成！${NC}"
