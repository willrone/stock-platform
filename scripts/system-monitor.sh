#!/bin/bash

# 系统监控脚本
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

# 配置
BACKEND_URL="http://localhost:8000"
FRONTEND_URL="http://localhost:3000"
PROMETHEUS_URL="http://localhost:9090"
GRAFANA_URL="http://localhost:3001"
TIMEOUT=10

# 检查服务健康状态
check_service_health() {
    local service_name=$1
    local url=$2
    local endpoint=${3:-"/health"}
    
    log_info "检查 $service_name 服务状态..."
    
    if curl -f -s --max-time $TIMEOUT "$url$endpoint" > /dev/null 2>&1; then
        log_success "$service_name 服务运行正常"
        return 0
    else
        log_error "$service_name 服务不可用"
        return 1
    fi
}

# 检查Docker容器状态
check_docker_containers() {
    log_info "检查Docker容器状态..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装"
        return 1
    fi
    
    local containers=$(docker ps --filter "name=stock-prediction" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}")
    
    if [ -z "$containers" ]; then
        log_warning "未找到股票预测平台相关容器"
        return 1
    fi
    
    echo "$containers"
    
    # 检查容器健康状态
    local unhealthy_containers=$(docker ps --filter "name=stock-prediction" --filter "health=unhealthy" --format "{{.Names}}")
    
    if [ -n "$unhealthy_containers" ]; then
        log_error "发现不健康的容器: $unhealthy_containers"
        return 1
    fi
    
    log_success "所有容器运行正常"
    return 0
}

# 检查系统资源
check_system_resources() {
    log_info "检查系统资源使用情况..."
    
    # CPU使用率
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
    log_info "CPU使用率: $cpu_usage%"
    
    # 内存使用率
    local memory_info=$(free -m | awk 'NR==2{printf "%.1f%%", $3*100/$2}')
    log_info "内存使用率: $memory_info"
    
    # 磁盘使用率
    log_info "磁盘使用情况:"
    df -h | grep -E "/$|/data|/var" | while read -r line; do
        echo "  $line"
        
        # 检查磁盘使用率是否超过80%
        usage=$(echo "$line" | awk '{print $5}' | sed 's/%//')
        if [ "$usage" -gt 80 ]; then
            log_warning "磁盘使用率过高: $usage%"
        fi
    done
    
    # 检查负载平均值
    local load_avg=$(uptime | awk -F'load average:' '{print $2}')
    log_info "系统负载:$load_avg"
}

# 检查网络连接
check_network_connectivity() {
    log_info "检查网络连接..."
    
    # 检查远端数据服务
    local remote_service="192.168.3.62:8080"
    if timeout 5 bash -c "</dev/tcp/192.168.3.62/8080" 2>/dev/null; then
        log_success "远端数据服务连接正常: $remote_service"
    else
        log_warning "远端数据服务连接失败: $remote_service"
    fi
    
    # 检查DNS解析
    if nslookup google.com > /dev/null 2>&1; then
        log_success "DNS解析正常"
    else
        log_warning "DNS解析异常"
    fi
}

# 检查日志错误
check_log_errors() {
    log_info "检查最近的日志错误..."
    
    local log_dir="./backend/data/logs"
    local error_log="$log_dir/error.log"
    
    if [ -f "$error_log" ]; then
        local recent_errors=$(tail -100 "$error_log" | grep -c "ERROR" 2>/dev/null || echo 0)
        local critical_errors=$(tail -100 "$error_log" | grep -c "CRITICAL" 2>/dev/null || echo 0)
        
        log_info "最近100行错误数量: $recent_errors"
        log_info "最近100行严重错误数量: $critical_errors"
        
        if [ "$critical_errors" -gt 0 ]; then
            log_error "发现严重错误，最近的错误:"
            tail -20 "$error_log" | grep "CRITICAL" | tail -3 | sed 's/^/  /'
        fi
        
        if [ "$recent_errors" -gt 10 ]; then
            log_warning "错误数量较多，请检查日志文件"
        fi
    else
        log_info "未找到错误日志文件"
    fi
}

# 检查数据库状态
check_database_status() {
    log_info "检查数据库状态..."
    
    local db_file="./backend/data/stock_prediction.db"
    
    if [ -f "$db_file" ]; then
        local db_size=$(du -h "$db_file" | cut -f1)
        log_info "数据库文件大小: $db_size"
        
        # 检查数据库是否可访问
        if command -v sqlite3 &> /dev/null; then
            if sqlite3 "$db_file" "SELECT COUNT(*) FROM sqlite_master;" > /dev/null 2>&1; then
                log_success "数据库连接正常"
            else
                log_error "数据库连接失败"
            fi
        else
            log_info "sqlite3未安装，跳过数据库连接测试"
        fi
    else
        log_warning "数据库文件不存在: $db_file"
    fi
}

# 检查数据文件
check_data_files() {
    log_info "检查数据文件状态..."
    
    local data_dir="./data/stocks"
    
    if [ -d "$data_dir" ]; then
        local file_count=$(find "$data_dir" -name "*.parquet" -type f | wc -l)
        local total_size=$(du -sh "$data_dir" 2>/dev/null | cut -f1 || echo "0")
        
        log_info "Parquet文件数量: $file_count"
        log_info "数据目录总大小: $total_size"
        
        # 检查最近更新的文件
        local recent_files=$(find "$data_dir" -name "*.parquet" -type f -mtime -1 | wc -l)
        log_info "最近24小时更新的文件: $recent_files"
        
        if [ "$file_count" -eq 0 ]; then
            log_warning "未找到数据文件"
        fi
    else
        log_warning "数据目录不存在: $data_dir"
    fi
}

# 生成监控报告
generate_monitor_report() {
    local report_file="./monitoring_report_$(date +%Y%m%d_%H%M%S).txt"
    
    log_info "生成监控报告: $report_file"
    
    {
        echo "=========================================="
        echo "系统监控报告 - $(date)"
        echo "=========================================="
        echo ""
        
        echo "服务状态检查:"
        check_service_health "后端API" "$BACKEND_URL" "/api/v1/health" 2>&1 | sed 's/^/  /'
        check_service_health "前端应用" "$FRONTEND_URL" "/api/health" 2>&1 | sed 's/^/  /'
        echo ""
        
        echo "Docker容器状态:"
        check_docker_containers 2>&1 | sed 's/^/  /'
        echo ""
        
        echo "系统资源使用:"
        check_system_resources 2>&1 | sed 's/^/  /'
        echo ""
        
        echo "网络连接检查:"
        check_network_connectivity 2>&1 | sed 's/^/  /'
        echo ""
        
        echo "数据库状态:"
        check_database_status 2>&1 | sed 's/^/  /'
        echo ""
        
        echo "数据文件状态:"
        check_data_files 2>&1 | sed 's/^/  /'
        echo ""
        
        echo "日志错误检查:"
        check_log_errors 2>&1 | sed 's/^/  /'
        echo ""
        
        echo "系统信息:"
        echo "  主机名: $(hostname)"
        echo "  操作系统: $(uname -a)"
        echo "  运行时间: $(uptime)"
        echo ""
        
    } > "$report_file"
    
    log_success "监控报告已生成: $report_file"
    
    # 显示报告摘要
    echo ""
    log_info "报告摘要:"
    head -30 "$report_file" | tail -20
}

# 实时监控模式
real_time_monitor() {
    log_info "启动实时监控模式 (按Ctrl+C退出)..."
    
    while true; do
        clear
        echo "========================================"
        echo "实时系统监控 - $(date)"
        echo "========================================"
        echo ""
        
        # 服务状态
        echo "服务状态:"
        check_service_health "后端" "$BACKEND_URL" "/api/v1/health" 2>/dev/null && echo "  ✓ 后端服务正常" || echo "  ✗ 后端服务异常"
        check_service_health "前端" "$FRONTEND_URL" "/api/health" 2>/dev/null && echo "  ✓ 前端服务正常" || echo "  ✗ 前端服务异常"
        echo ""
        
        # 系统资源
        echo "系统资源:"
        local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//' || echo "N/A")
        local memory_usage=$(free -m | awk 'NR==2{printf "%.1f%%", $3*100/$2}' || echo "N/A")
        local load_avg=$(uptime | awk -F'load average:' '{print $2}' || echo "N/A")
        
        echo "  CPU使用率: $cpu_usage%"
        echo "  内存使用率: $memory_usage"
        echo "  系统负载:$load_avg"
        echo ""
        
        # Docker容器
        echo "Docker容器:"
        if command -v docker &> /dev/null; then
            docker ps --filter "name=stock-prediction" --format "  {{.Names}}: {{.Status}}" 2>/dev/null || echo "  无容器运行"
        else
            echo "  Docker未安装"
        fi
        echo ""
        
        echo "下次更新: $(date -d '+30 seconds')"
        sleep 30
    done
}

# 主函数
main() {
    local action=${1:-"check"}
    
    case "$action" in
        "health")
            echo "========================================"
            echo "    服务健康检查"
            echo "========================================"
            echo ""
            check_service_health "后端API" "$BACKEND_URL" "/api/v1/health"
            check_service_health "前端应用" "$FRONTEND_URL" "/api/health"
            if command -v curl &> /dev/null; then
                check_service_health "Prometheus" "$PROMETHEUS_URL" "/-/healthy" 2>/dev/null || log_info "Prometheus未运行或不可访问"
                check_service_health "Grafana" "$GRAFANA_URL" "/api/health" 2>/dev/null || log_info "Grafana未运行或不可访问"
            fi
            ;;
        "docker")
            echo "========================================"
            echo "    Docker容器检查"
            echo "========================================"
            echo ""
            check_docker_containers
            ;;
        "system")
            echo "========================================"
            echo "    系统资源检查"
            echo "========================================"
            echo ""
            check_system_resources
            ;;
        "network")
            echo "========================================"
            echo "    网络连接检查"
            echo "========================================"
            echo ""
            check_network_connectivity
            ;;
        "logs")
            echo "========================================"
            echo "    日志错误检查"
            echo "========================================"
            echo ""
            check_log_errors
            ;;
        "data")
            echo "========================================"
            echo "    数据状态检查"
            echo "========================================"
            echo ""
            check_database_status
            check_data_files
            ;;
        "report")
            generate_monitor_report
            ;;
        "monitor")
            real_time_monitor
            ;;
        "check"|"all")
            echo "========================================"
            echo "    完整系统检查"
            echo "========================================"
            echo ""
            check_service_health "后端API" "$BACKEND_URL" "/api/v1/health"
            check_service_health "前端应用" "$FRONTEND_URL" "/api/health"
            echo ""
            check_docker_containers
            echo ""
            check_system_resources
            echo ""
            check_network_connectivity
            echo ""
            check_database_status
            echo ""
            check_data_files
            echo ""
            check_log_errors
            ;;
        "help"|"-h"|"--help")
            echo "用法: $0 [action]"
            echo ""
            echo "操作:"
            echo "  health    - 检查服务健康状态"
            echo "  docker    - 检查Docker容器状态"
            echo "  system    - 检查系统资源使用"
            echo "  network   - 检查网络连接"
            echo "  logs      - 检查日志错误"
            echo "  data      - 检查数据状态"
            echo "  report    - 生成完整监控报告"
            echo "  monitor   - 实时监控模式"
            echo "  check     - 执行所有检查 (默认)"
            echo "  help      - 显示帮助信息"
            ;;
        *)
            log_error "未知操作: $action"
            echo "使用 '$0 help' 查看帮助信息"
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"