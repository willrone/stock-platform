#!/bin/bash

# 部署验证测试脚本
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

# 测试计数器
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# 运行测试
run_test() {
    local test_name=$1
    local test_command=$2
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    log_info "运行测试: $test_name"
    
    if eval "$test_command" > /dev/null 2>&1; then
        log_success "✓ $test_name"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        log_error "✗ $test_name"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# 测试Docker配置文件
test_docker_configs() {
    log_info "测试Docker配置文件..."
    
    run_test "后端Dockerfile存在" "[ -f backend/Dockerfile ]"
    run_test "前端Dockerfile存在" "[ -f frontend/Dockerfile ]"
    run_test "前端开发Dockerfile存在" "[ -f frontend/Dockerfile.dev ]"
    run_test "Docker Compose生产配置存在" "[ -f docker-compose.yml ]"
    run_test "Docker Compose开发配置存在" "[ -f docker-compose.dev.yml ]"
    
    # 验证Docker Compose配置语法
    if command -v docker-compose &> /dev/null; then
        run_test "Docker Compose生产配置语法正确" "docker-compose -f docker-compose.yml config"
        run_test "Docker Compose开发配置语法正确" "docker-compose -f docker-compose.dev.yml config"
    else
        log_warning "Docker Compose未安装，跳过语法检查"
    fi
}

# 测试环境配置
test_environment_configs() {
    log_info "测试环境配置..."
    
    run_test "环境变量示例文件存在" "[ -f .env.example ]"
    run_test "Nginx配置文件存在" "[ -f nginx/nginx.conf ]"
    run_test "系统服务配置存在" "[ -f systemd/stock-prediction.service ]"
    
    # 检查环境变量文件内容
    if [ -f .env.example ]; then
        run_test "环境变量包含数据库配置" "grep -q 'DATABASE_URL' .env.example"
        run_test "环境变量包含API配置" "grep -q 'NEXT_PUBLIC_API_URL' .env.example"
        run_test "环境变量包含日志配置" "grep -q 'LOG_LEVEL' .env.example"
    fi
}

# 测试启动脚本
test_startup_scripts() {
    log_info "测试启动脚本..."
    
    run_test "启动脚本存在" "[ -f scripts/start.sh ]"
    run_test "停止脚本存在" "[ -f scripts/stop.sh ]"
    run_test "启动脚本可执行" "[ -x scripts/start.sh ]"
    run_test "停止脚本可执行" "[ -x scripts/stop.sh ]"
    
    # 检查脚本语法
    run_test "启动脚本语法正确" "bash -n scripts/start.sh"
    run_test "停止脚本语法正确" "bash -n scripts/stop.sh"
}

# 测试监控配置
test_monitoring_configs() {
    log_info "测试监控配置..."
    
    run_test "Prometheus配置存在" "[ -f monitoring/prometheus.yml ]"
    run_test "Grafana数据源配置存在" "[ -f monitoring/grafana/datasources/prometheus.yml ]"
    run_test "系统监控仪表板存在" "[ -f monitoring/grafana/dashboards/system-overview.json ]"
    run_test "应用性能仪表板存在" "[ -f monitoring/grafana/dashboards/application-performance.json ]"
    run_test "业务指标仪表板存在" "[ -f monitoring/grafana/dashboards/business-metrics.json ]"
    run_test "告警规则存在" "[ -f monitoring/rules/alerts.yml ]"
    
    # 验证JSON配置文件语法
    if command -v jq &> /dev/null; then
        run_test "系统监控仪表板JSON语法正确" "jq empty monitoring/grafana/dashboards/system-overview.json"
        run_test "应用性能仪表板JSON语法正确" "jq empty monitoring/grafana/dashboards/application-performance.json"
        run_test "业务指标仪表板JSON语法正确" "jq empty monitoring/grafana/dashboards/business-metrics.json"
    else
        log_warning "jq未安装，跳过JSON语法检查"
    fi
}

# 测试日志配置
test_logging_configs() {
    log_info "测试日志配置..."
    
    run_test "日志管理脚本存在" "[ -f scripts/log-management.sh ]"
    run_test "日志管理脚本可执行" "[ -x scripts/log-management.sh ]"
    run_test "日志管理脚本语法正确" "bash -n scripts/log-management.sh"
    
    # 检查后端日志配置
    run_test "后端日志配置文件存在" "[ -f backend/app/core/logging.py ]"
    run_test "后端指标收集模块存在" "[ -f backend/app/core/metrics.py ]"
}

# 测试管理脚本
test_management_scripts() {
    log_info "测试管理脚本..."
    
    run_test "系统监控脚本存在" "[ -f scripts/system-monitor.sh ]"
    run_test "系统监控脚本可执行" "[ -x scripts/system-monitor.sh ]"
    run_test "系统监控脚本语法正确" "bash -n scripts/system-monitor.sh"
    
    run_test "监控设置脚本存在" "[ -f monitoring/setup-monitoring.sh ]"
    run_test "监控设置脚本可执行" "[ -x monitoring/setup-monitoring.sh ]"
    run_test "监控设置脚本语法正确" "bash -n monitoring/setup-monitoring.sh"
}

# 测试前端API端点
test_frontend_api_endpoints() {
    log_info "测试前端API端点..."
    
    run_test "前端健康检查端点存在" "[ -f frontend/src/app/api/health/route.ts ]"
    run_test "前端指标端点存在" "[ -f frontend/src/app/api/metrics/route.ts ]"
    
    # 检查TypeScript语法（如果安装了tsc）
    if command -v tsc &> /dev/null; then
        run_test "前端健康检查端点语法正确" "cd frontend && tsc --noEmit src/app/api/health/route.ts"
        run_test "前端指标端点语法正确" "cd frontend && tsc --noEmit src/app/api/metrics/route.ts"
    else
        log_warning "TypeScript编译器未安装，跳过语法检查"
    fi
}

# 测试文档
test_documentation() {
    log_info "测试文档..."
    
    run_test "部署文档存在" "[ -f DEPLOYMENT.md ]"
    run_test "部署文档不为空" "[ -s DEPLOYMENT.md ]"
    
    # 检查文档内容
    if [ -f DEPLOYMENT.md ]; then
        run_test "部署文档包含系统要求" "grep -q '系统要求' DEPLOYMENT.md"
        run_test "部署文档包含快速开始" "grep -q '快速开始' DEPLOYMENT.md"
        run_test "部署文档包含故障排除" "grep -q '故障排除' DEPLOYMENT.md"
    fi
}

# 测试目录结构
test_directory_structure() {
    log_info "测试目录结构..."
    
    local required_dirs=(
        "backend"
        "frontend"
        "scripts"
        "monitoring"
        "monitoring/grafana"
        "monitoring/grafana/dashboards"
        "monitoring/grafana/datasources"
        "monitoring/rules"
        "nginx"
        "systemd"
    )
    
    for dir in "${required_dirs[@]}"; do
        run_test "目录存在: $dir" "[ -d $dir ]"
    done
}

# 模拟部署测试
test_deployment_simulation() {
    log_info "模拟部署测试..."
    
    # 检查Docker是否可用
    if command -v docker &> /dev/null; then
        run_test "Docker服务运行中" "docker info"
        
        # 尝试构建镜像（不实际构建，只检查Dockerfile）
        run_test "后端镜像可构建" "docker build --dry-run backend/ -f backend/Dockerfile"
        run_test "前端镜像可构建" "docker build --dry-run frontend/ -f frontend/Dockerfile"
    else
        log_warning "Docker未安装，跳过部署模拟测试"
    fi
}

# 生成测试报告
generate_test_report() {
    local report_file="deployment_test_report_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "=========================================="
        echo "部署验证测试报告 - $(date)"
        echo "=========================================="
        echo ""
        echo "测试统计:"
        echo "  总测试数: $TOTAL_TESTS"
        echo "  通过测试: $PASSED_TESTS"
        echo "  失败测试: $FAILED_TESTS"
        echo "  成功率: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%"
        echo ""
        
        if [ $FAILED_TESTS -eq 0 ]; then
            echo "✓ 所有测试通过！部署配置验证成功。"
        else
            echo "✗ 有 $FAILED_TESTS 个测试失败，请检查配置。"
        fi
        echo ""
        
        echo "测试环境信息:"
        echo "  操作系统: $(uname -a)"
        echo "  Docker版本: $(docker --version 2>/dev/null || echo '未安装')"
        echo "  Docker Compose版本: $(docker-compose --version 2>/dev/null || echo '未安装')"
        echo "  当前目录: $(pwd)"
        echo ""
        
    } > "$report_file"
    
    log_success "测试报告已生成: $report_file"
    
    # 显示报告内容
    cat "$report_file"
}

# 主函数
main() {
    echo "========================================"
    echo "    部署配置验证测试"
    echo "========================================"
    echo ""
    
    # 运行所有测试
    test_directory_structure
    test_docker_configs
    test_environment_configs
    test_startup_scripts
    test_monitoring_configs
    test_logging_configs
    test_management_scripts
    test_frontend_api_endpoints
    test_documentation
    test_deployment_simulation
    
    echo ""
    echo "========================================"
    echo "    测试结果汇总"
    echo "========================================"
    echo ""
    
    generate_test_report
    
    # 返回适当的退出码
    if [ $FAILED_TESTS -eq 0 ]; then
        log_success "所有部署配置验证通过！"
        exit 0
    else
        log_error "部署配置验证失败，请修复问题后重试"
        exit 1
    fi
}

# 执行主函数
main "$@"