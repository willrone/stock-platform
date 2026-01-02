#!/bin/bash

# MLOps系统状态检查脚本
# 用于检查MLOps系统的运行状态

set -e

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

# 检查服务状态
check_service_status() {
    log_info "检查服务状态..."
    
    # 检查后端服务
    if pgrep -f "uvicorn.*main:app" > /dev/null; then
        log_success "后端服务运行中"
        
        # 检查API健康状态
        if curl -s http://localhost:8000/health > /dev/null; then
            log_success "后端API响应正常"
        else
            log_error "后端API无响应"
        fi
    else
        log_error "后端服务未运行"
    fi
    
    # 检查前端服务
    if pgrep -f "npm.*run.*dev" > /dev/null; then
        log_success "前端服务运行中"
    else
        log_warning "前端服务未运行"
    fi
    
    # 检查systemd服务
    if systemctl is-active --quiet mlops-backend 2>/dev/null; then
        log_success "MLOps后端系统服务运行中"
    else
        log_warning "MLOps后端系统服务未运行"
    fi
}

# 检查系统资源
check_system_resources() {
    log_info "检查系统资源..."
    
    # CPU使用率
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')
    if (( $(echo "$cpu_usage > 80" | bc -l) )); then
        log_warning "CPU使用率较高: ${cpu_usage}%"
    else
        log_success "CPU使用率正常: ${cpu_usage}%"
    fi
    
    # 内存使用率
    memory_info=$(free | grep Mem)
    total_mem=$(echo $memory_info | awk '{print $2}')
    used_mem=$(echo $memory_info | awk '{print $3}')
    memory_usage=$(echo "scale=1; $used_mem * 100 / $total_mem" | bc)
    
    if (( $(echo "$memory_usage > 85" | bc -l) )); then
        log_warning "内存使用率较高: ${memory_usage}%"
    else
        log_success "内存使用率正常: ${memory_usage}%"
    fi
    
    # 磁盘使用率
    disk_usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ "$disk_usage" -gt 90 ]; then
        log_warning "磁盘使用率较高: ${disk_usage}%"
    else
        log_success "磁盘使用率正常: ${disk_usage}%"
    fi
}

# 检查数据库状态
check_database_status() {
    log_info "检查数据库状态..."
    
    if [ -f "backend/data/app.db" ]; then
        db_size=$(du -h backend/data/app.db | cut -f1)
        log_success "数据库文件存在，大小: $db_size"
        
        # 检查数据库连接
        cd backend
        if [ -d "venv" ]; then
            source venv/bin/activate
            python -c "
from app.core.database import SessionLocal
try:
    session = SessionLocal()
    session.execute('SELECT 1')
    session.close()
    print('数据库连接正常')
except Exception as e:
    print(f'数据库连接失败: {e}')
    exit(1)
" && log_success "数据库连接正常" || log_error "数据库连接失败"
        else
            log_warning "Python虚拟环境不存在"
        fi
        cd ..
    else
        log_error "数据库文件不存在"
    fi
}

# 检查MLOps组件状态
check_mlops_components() {
    log_info "检查MLOps组件状态..."
    
    cd backend
    if [ -d "venv" ]; then
        source venv/bin/activate
        
        # 检查关键模块
        python -c "
import sys
import importlib

components = [
    ('特征工程', 'app.services.features.feature_pipeline'),
    ('Qlib集成', 'app.services.qlib.unified_qlib_training_engine'),
    ('模型管理', 'app.services.models.model_lifecycle_manager'),
    ('监控系统', 'app.services.monitoring.performance_monitor'),
    ('错误处理', 'app.services.system.error_handler'),
    ('A/B测试', 'app.services.ab_testing.traffic_manager'),
    ('数据版本控制', 'app.services.data_versioning.version_manager')
]

for name, module in components:
    try:
        importlib.import_module(module)
        print(f'✓ {name}: 正常')
    except ImportError as e:
        print(f'✗ {name}: 导入失败 - {e}')
    except Exception as e:
        print(f'? {name}: 未知错误 - {e}')
"
    else
        log_warning "Python虚拟环境不存在，跳过组件检查"
    fi
    cd ..
}

# 检查日志文件
check_log_files() {
    log_info "检查日志文件..."
    
    log_files=(
        "logs/backend.log"
        "backend/logs/mlops.log"
        "logs/frontend.log"
    )
    
    for log_file in "${log_files[@]}"; do
        if [ -f "$log_file" ]; then
            file_size=$(du -h "$log_file" | cut -f1)
            log_success "$log_file 存在，大小: $file_size"
            
            # 检查最近的错误
            error_count=$(grep -c "ERROR" "$log_file" 2>/dev/null || echo "0")
            if [ "$error_count" -gt 0 ]; then
                log_warning "$log_file 包含 $error_count 个错误"
            fi
        else
            log_warning "$log_file 不存在"
        fi
    done
}

# 检查配置文件
check_config_files() {
    log_info "检查配置文件..."
    
    config_files=(
        ".env"
        "backend/config/mlops_config.yaml"
    )
    
    for config_file in "${config_files[@]}"; do
        if [ -f "$config_file" ]; then
            log_success "$config_file 存在"
        else
            log_warning "$config_file 不存在"
        fi
    done
}

# 检查数据目录
check_data_directories() {
    log_info "检查数据目录..."
    
    data_dirs=(
        "backend/data/models"
        "backend/data/features"
        "backend/data/qlib_cache"
        "data/backups"
    )
    
    for data_dir in "${data_dirs[@]}"; do
        if [ -d "$data_dir" ]; then
            file_count=$(find "$data_dir" -type f | wc -l)
            dir_size=$(du -sh "$data_dir" | cut -f1)
            log_success "$data_dir 存在，包含 $file_count 个文件，大小: $dir_size"
        else
            log_warning "$data_dir 不存在"
        fi
    done
}

# 检查网络连接
check_network_connectivity() {
    log_info "检查网络连接..."
    
    # 检查本地端口
    if netstat -tuln | grep -q ":8000"; then
        log_success "后端端口 8000 正在监听"
    else
        log_error "后端端口 8000 未监听"
    fi
    
    if netstat -tuln | grep -q ":3000"; then
        log_success "前端端口 3000 正在监听"
    else
        log_warning "前端端口 3000 未监听"
    fi
    
    # 检查外部连接
    if ping -c 1 google.com > /dev/null 2>&1; then
        log_success "外部网络连接正常"
    else
        log_warning "外部网络连接异常"
    fi
}

# 生成状态报告
generate_status_report() {
    log_info "生成状态报告..."
    
    report_file="logs/status_report_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "MLOps系统状态报告"
        echo "生成时间: $(date)"
        echo "================================"
        echo
        
        echo "系统信息:"
        echo "  操作系统: $(uname -s)"
        echo "  内核版本: $(uname -r)"
        echo "  主机名: $(hostname)"
        echo "  运行时间: $(uptime -p)"
        echo
        
        echo "服务状态:"
        if pgrep -f "uvicorn.*main:app" > /dev/null; then
            echo "  后端服务: 运行中"
        else
            echo "  后端服务: 未运行"
        fi
        
        if pgrep -f "npm.*run.*dev" > /dev/null; then
            echo "  前端服务: 运行中"
        else
            echo "  前端服务: 未运行"
        fi
        echo
        
        echo "资源使用:"
        echo "  CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}')"
        echo "  内存: $(free -h | grep Mem | awk '{print $3 "/" $2}')"
        echo "  磁盘: $(df -h / | awk 'NR==2 {print $3 "/" $2 " (" $5 ")"}')"
        echo
        
        echo "进程信息:"
        ps aux | grep -E "(uvicorn|npm)" | grep -v grep
        echo
        
        echo "端口监听:"
        netstat -tuln | grep -E ":(8000|3000)"
        echo
        
    } > "$report_file"
    
    log_success "状态报告已生成: $report_file"
}

# 显示快速状态
show_quick_status() {
    echo "MLOps系统快速状态检查"
    echo "======================"
    
    # 服务状态
    if pgrep -f "uvicorn.*main:app" > /dev/null; then
        echo -e "后端服务: ${GREEN}运行中${NC}"
    else
        echo -e "后端服务: ${RED}未运行${NC}"
    fi
    
    if pgrep -f "npm.*run.*dev" > /dev/null; then
        echo -e "前端服务: ${GREEN}运行中${NC}"
    else
        echo -e "前端服务: ${YELLOW}未运行${NC}"
    fi
    
    # API状态
    if curl -s http://localhost:8000/health > /dev/null; then
        echo -e "API状态: ${GREEN}正常${NC}"
    else
        echo -e "API状态: ${RED}异常${NC}"
    fi
    
    # 资源状态
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')
    memory_usage=$(free | awk 'NR==2{printf "%.1f", $3*100/$2}')
    
    echo "CPU使用率: ${cpu_usage}%"
    echo "内存使用率: ${memory_usage}%"
}

# 主函数
main() {
    case "${1:-full}" in
        "quick"|"-q"|"--quick")
            show_quick_status
            ;;
        "full"|"-f"|"--full"|"")
            log_info "开始MLOps系统状态检查..."
            check_service_status
            check_system_resources
            check_database_status
            check_mlops_components
            check_log_files
            check_config_files
            check_data_directories
            check_network_connectivity
            generate_status_report
            log_success "状态检查完成"
            ;;
        "report"|"-r"|"--report")
            generate_status_report
            ;;
        "-h"|"--help")
            echo "MLOps状态检查脚本"
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  quick, -q, --quick    快速状态检查"
            echo "  full, -f, --full      完整状态检查 (默认)"
            echo "  report, -r, --report  仅生成状态报告"
            echo "  -h, --help            显示帮助信息"
            ;;
        *)
            log_error "未知选项: $1"
            echo "使用 -h 或 --help 查看帮助信息"
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"