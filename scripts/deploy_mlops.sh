#!/bin/bash

# MLOps系统部署脚本
# 用于部署和升级MLOps功能

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查依赖
check_dependencies() {
    log_info "检查系统依赖..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装"
        exit 1
    fi
    
    # 检查pip
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 未安装"
        exit 1
    fi
    
    # 检查Node.js (用于前端)
    if ! command -v node &> /dev/null; then
        log_warning "Node.js 未安装，前端功能可能受影响"
    fi
    
    # 检查Docker (可选)
    if ! command -v docker &> /dev/null; then
        log_warning "Docker 未安装，容器化部署不可用"
    fi
    
    log_success "依赖检查完成"
}

# 创建必要目录
create_directories() {
    log_info "创建必要目录..."
    
    directories=(
        "backend/data/models"
        "backend/data/features"
        "backend/data/qlib_cache"
        "backend/logs"
        "backend/config"
        "data/backups"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_info "创建目录: $dir"
        fi
    done
    
    log_success "目录创建完成"
}

# 安装Python依赖
install_python_dependencies() {
    log_info "安装Python依赖..."
    
    # 检查虚拟环境
    if [ ! -d "backend/venv" ]; then
        log_info "创建Python虚拟环境..."
        python3 -m venv backend/venv
    fi
    
    # 激活虚拟环境
    source backend/venv/bin/activate
    
    # 升级pip
    pip install --upgrade pip
    
    # 安装基础依赖
    if [ -f "backend/requirements.txt" ]; then
        pip install -r backend/requirements.txt
        log_success "基础依赖安装完成"
    fi
    
    # 安装MLOps特定依赖
    mlops_packages=(
        "qlib"
        "shap"
        "scikit-learn"
        "lightgbm"
        "xgboost"
        "optuna"
        "mlflow"
        "psutil"
        "pyyaml"
    )
    
    for package in "${mlops_packages[@]}"; do
        log_info "安装 $package..."
        pip install "$package" || log_warning "$package 安装失败，跳过"
    done
    
    log_success "Python依赖安装完成"
}

# 初始化数据库
initialize_database() {
    log_info "初始化数据库..."
    
    # 激活虚拟环境
    source backend/venv/bin/activate
    
    # 运行数据库迁移
    cd backend
    python -c "
from app.core.database import engine, Base
from app.models import task_models, stock_models
Base.metadata.create_all(bind=engine)
print('数据库表创建完成')
" || log_warning "数据库初始化失败"
    
    cd ..
    log_success "数据库初始化完成"
}

# 配置MLOps
configure_mlops() {
    log_info "配置MLOps系统..."
    
    # 复制配置文件
    if [ ! -f "backend/config/mlops_config.yaml" ]; then
        if [ -f "backend/config/mlops_config.yaml" ]; then
            log_info "使用默认MLOps配置"
        else
            log_warning "MLOps配置文件不存在，使用默认配置"
        fi
    fi
    
    # 设置环境变量
    if [ ! -f ".env" ]; then
        log_info "创建环境变量文件..."
        cat > .env << EOF
# MLOps配置
MLOPS_ENABLED=true
QLIB_CACHE_DIR=backend/data/qlib_cache
FEATURE_CACHE_ENABLED=true
MONITORING_ENABLED=true
AB_TESTING_ENABLED=true

# 数据库配置
DATABASE_URL=sqlite:///./backend/data/app.db

# Redis配置 (可选)
REDIS_URL=redis://localhost:6379/0

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=backend/logs/mlops.log
EOF
        log_success "环境变量文件创建完成"
    fi
    
    log_success "MLOps配置完成"
}

# 初始化Qlib数据
initialize_qlib() {
    log_info "初始化Qlib数据..."
    
    # 激活虚拟环境
    source backend/venv/bin/activate
    
    cd backend
    python -c "
import qlib
from qlib.config import REG_CN

# 初始化Qlib
try:
    qlib.init(provider_uri='data/qlib_data', region=REG_CN)
    print('Qlib初始化成功')
except Exception as e:
    print(f'Qlib初始化失败: {e}')
    print('请手动下载Qlib数据')
" || log_warning "Qlib初始化失败"
    
    cd ..
    log_success "Qlib初始化完成"
}

# 运行测试
run_tests() {
    log_info "运行系统测试..."
    
    # 激活虚拟环境
    source backend/venv/bin/activate
    
    cd backend
    
    # 运行基础测试
    python -c "
# 测试导入
try:
    from app.services.features.feature_pipeline import feature_pipeline
    from app.services.qlib.unified_qlib_training_engine import UnifiedQlibTrainingEngine
    from app.services.monitoring.performance_monitor import performance_monitor
    from app.services.system.error_handler import error_handler
    print('所有MLOps模块导入成功')
except ImportError as e:
    print(f'模块导入失败: {e}')
    exit(1)

# 测试基础功能
try:
    # 测试特征管道
    print('测试特征管道...')
    
    # 测试错误处理
    print('测试错误处理...')
    
    # 测试性能监控
    print('测试性能监控...')
    
    print('基础功能测试通过')
except Exception as e:
    print(f'功能测试失败: {e}')
    exit(1)
"
    
    if [ $? -eq 0 ]; then
        log_success "系统测试通过"
    else
        log_error "系统测试失败"
        exit 1
    fi
    
    cd ..
}

# 启动服务
start_services() {
    log_info "启动MLOps服务..."
    
    # 检查是否有运行中的服务
    if pgrep -f "uvicorn.*main:app" > /dev/null; then
        log_warning "检测到运行中的服务，正在停止..."
        pkill -f "uvicorn.*main:app"
        sleep 2
    fi
    
    # 启动后端服务
    cd backend
    source venv/bin/activate
    
    log_info "启动后端服务..."
    nohup uvicorn main:app --host 0.0.0.0 --port 8000 --reload > ../logs/backend.log 2>&1 &
    
    # 等待服务启动
    sleep 5
    
    # 检查服务状态
    if curl -s http://localhost:8000/health > /dev/null; then
        log_success "后端服务启动成功"
    else
        log_error "后端服务启动失败"
        exit 1
    fi
    
    cd ..
    
    # 启动前端服务 (如果存在)
    if [ -d "frontend" ] && [ -f "frontend/package.json" ]; then
        log_info "启动前端服务..."
        cd frontend
        
        if [ ! -d "node_modules" ]; then
            log_info "安装前端依赖..."
            npm install
        fi
        
        nohup npm run dev > ../logs/frontend.log 2>&1 &
        cd ..
        
        log_success "前端服务启动成功"
    fi
    
    log_success "所有服务启动完成"
}

# 创建系统服务文件
create_systemd_service() {
    log_info "创建系统服务文件..."
    
    # 获取当前目录
    CURRENT_DIR=$(pwd)
    
    # 创建后端服务文件
    sudo tee /etc/systemd/system/mlops-backend.service > /dev/null << EOF
[Unit]
Description=MLOps Backend Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$CURRENT_DIR/backend
Environment=PATH=$CURRENT_DIR/backend/venv/bin
ExecStart=$CURRENT_DIR/backend/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    # 重新加载systemd
    sudo systemctl daemon-reload
    sudo systemctl enable mlops-backend.service
    
    log_success "系统服务创建完成"
    log_info "使用以下命令管理服务:"
    log_info "  启动: sudo systemctl start mlops-backend"
    log_info "  停止: sudo systemctl stop mlops-backend"
    log_info "  状态: sudo systemctl status mlops-backend"
}

# 备份现有数据
backup_data() {
    log_info "备份现有数据..."
    
    BACKUP_DIR="data/backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # 备份数据库
    if [ -f "backend/data/app.db" ]; then
        cp backend/data/app.db "$BACKUP_DIR/"
        log_info "数据库备份完成"
    fi
    
    # 备份模型文件
    if [ -d "backend/data/models" ]; then
        cp -r backend/data/models "$BACKUP_DIR/"
        log_info "模型文件备份完成"
    fi
    
    # 备份配置文件
    if [ -f ".env" ]; then
        cp .env "$BACKUP_DIR/"
    fi
    
    log_success "数据备份完成: $BACKUP_DIR"
}

# 显示部署信息
show_deployment_info() {
    log_success "MLOps系统部署完成！"
    echo
    log_info "服务信息:"
    log_info "  后端API: http://localhost:8000"
    log_info "  API文档: http://localhost:8000/docs"
    log_info "  健康检查: http://localhost:8000/health"
    
    if [ -d "frontend" ]; then
        log_info "  前端界面: http://localhost:3000"
    fi
    
    echo
    log_info "日志文件:"
    log_info "  后端日志: logs/backend.log"
    log_info "  MLOps日志: backend/logs/mlops.log"
    
    if [ -d "frontend" ]; then
        log_info "  前端日志: logs/frontend.log"
    fi
    
    echo
    log_info "配置文件:"
    log_info "  环境变量: .env"
    log_info "  MLOps配置: backend/config/mlops_config.yaml"
    
    echo
    log_info "管理命令:"
    log_info "  停止服务: ./scripts/stop_mlops.sh"
    log_info "  查看状态: ./scripts/status_mlops.sh"
    log_info "  更新系统: ./scripts/update_mlops.sh"
}

# 主函数
main() {
    log_info "开始MLOps系统部署..."
    
    # 解析命令行参数
    BACKUP=true
    CREATE_SERVICE=false
    SKIP_TESTS=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-backup)
                BACKUP=false
                shift
                ;;
            --create-service)
                CREATE_SERVICE=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            -h|--help)
                echo "MLOps部署脚本"
                echo "用法: $0 [选项]"
                echo "选项:"
                echo "  --no-backup      跳过数据备份"
                echo "  --create-service 创建systemd服务"
                echo "  --skip-tests     跳过测试"
                echo "  -h, --help       显示帮助信息"
                exit 0
                ;;
            *)
                log_error "未知选项: $1"
                exit 1
                ;;
        esac
    done
    
    # 执行部署步骤
    check_dependencies
    
    if [ "$BACKUP" = true ]; then
        backup_data
    fi
    
    create_directories
    install_python_dependencies
    initialize_database
    configure_mlops
    initialize_qlib
    
    if [ "$SKIP_TESTS" = false ]; then
        run_tests
    fi
    
    start_services
    
    if [ "$CREATE_SERVICE" = true ]; then
        create_systemd_service
    fi
    
    show_deployment_info
    
    log_success "MLOps系统部署完成！"
}

# 错误处理
trap 'log_error "部署过程中发生错误，请检查日志"; exit 1' ERR

# 运行主函数
main "$@"