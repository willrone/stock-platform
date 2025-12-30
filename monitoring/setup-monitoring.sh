#!/bin/bash

# 监控系统设置脚本
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

# 检查依赖
check_dependencies() {
    log_info "检查依赖..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose未安装"
        exit 1
    fi
    
    log_success "依赖检查通过"
}

# 创建监控目录结构
create_monitoring_directories() {
    log_info "创建监控目录结构..."
    
    mkdir -p monitoring/prometheus/data
    mkdir -p monitoring/grafana/data
    mkdir -p monitoring/grafana/logs
    mkdir -p monitoring/alertmanager/data
    mkdir -p monitoring/node-exporter
    mkdir -p monitoring/cadvisor
    
    # 设置权限
    chmod 755 monitoring/prometheus/data
    chmod 755 monitoring/grafana/data
    chmod 755 monitoring/grafana/logs
    
    log_success "监控目录创建完成"
}

# 生成Prometheus配置
generate_prometheus_config() {
    log_info "生成Prometheus配置..."
    
    if [ ! -f "monitoring/prometheus.yml" ]; then
        log_warning "Prometheus配置文件不存在，使用默认配置"
        return 1
    fi
    
    # 验证配置文件
    if command -v promtool &> /dev/null; then
        if promtool check config monitoring/prometheus.yml; then
            log_success "Prometheus配置验证通过"
        else
            log_error "Prometheus配置验证失败"
            return 1
        fi
    else
        log_warning "promtool未安装，跳过配置验证"
    fi
    
    return 0
}

# 设置Grafana配置
setup_grafana_config() {
    log_info "设置Grafana配置..."
    
    # 创建Grafana配置文件
    cat > monitoring/grafana.ini << EOF
[server]
http_port = 3000
domain = localhost

[security]
admin_user = admin
admin_password = admin123

[users]
allow_sign_up = false

[auth.anonymous]
enabled = false

[dashboards]
default_home_dashboard_path = /etc/grafana/provisioning/dashboards/system-overview.json

[log]
mode = console file
level = info

[paths]
data = /var/lib/grafana
logs = /var/log/grafana
plugins = /var/lib/grafana/plugins
provisioning = /etc/grafana/provisioning
EOF

    log_success "Grafana配置生成完成"
}

# 创建Docker Compose监控配置
create_monitoring_compose() {
    log_info "创建监控Docker Compose配置..."
    
    cat > docker-compose.monitoring.yml << EOF
version: '3.8'

services:
  # Prometheus监控
  prometheus:
    image: prom/prometheus:latest
    container_name: stock-prediction-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./monitoring/rules:/etc/prometheus/rules:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
    networks:
      - monitoring
    restart: unless-stopped

  # Grafana仪表板
  grafana:
    image: grafana/grafana:latest
    container_name: stock-prediction-grafana
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana.ini:/etc/grafana/grafana.ini:ro
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro
    depends_on:
      - prometheus
    networks:
      - monitoring
    restart: unless-stopped

  # Node Exporter (系统指标)
  node-exporter:
    image: prom/node-exporter:latest
    container_name: stock-prediction-node-exporter
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    networks:
      - monitoring
    restart: unless-stopped

  # cAdvisor (容器指标)
  cadvisor:
    image: gcr.io/cadvisor/cadvisor:latest
    container_name: stock-prediction-cadvisor
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
      - /dev/disk/:/dev/disk:ro
    privileged: true
    devices:
      - /dev/kmsg
    networks:
      - monitoring
    restart: unless-stopped

  # AlertManager (告警管理)
  alertmanager:
    image: prom/alertmanager:latest
    container_name: stock-prediction-alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro
      - alertmanager_data:/alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
      - '--web.external-url=http://localhost:9093'
    networks:
      - monitoring
    restart: unless-stopped

networks:
  monitoring:
    driver: bridge

volumes:
  prometheus_data:
  grafana_data:
  alertmanager_data:
EOF

    log_success "监控Docker Compose配置创建完成"
}

# 创建AlertManager配置
create_alertmanager_config() {
    log_info "创建AlertManager配置..."
    
    cat > monitoring/alertmanager.yml << EOF
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@stock-prediction.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
- name: 'web.hook'
  webhook_configs:
  - url: 'http://localhost:5001/webhook'
    send_resolved: true

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'dev', 'instance']
EOF

    log_success "AlertManager配置创建完成"
}

# 启动监控服务
start_monitoring_services() {
    log_info "启动监控服务..."
    
    if [ -f "docker-compose.monitoring.yml" ]; then
        docker-compose -f docker-compose.monitoring.yml up -d
        log_success "监控服务启动完成"
    else
        log_error "监控Docker Compose文件不存在"
        return 1
    fi
}

# 检查监控服务状态
check_monitoring_services() {
    log_info "检查监控服务状态..."
    
    local services=("prometheus:9090" "grafana:3001" "node-exporter:9100" "cadvisor:8080")
    
    for service in "${services[@]}"; do
        local name=$(echo "$service" | cut -d':' -f1)
        local port=$(echo "$service" | cut -d':' -f2)
        
        if curl -f -s --max-time 10 "http://localhost:$port" > /dev/null 2>&1; then
            log_success "$name 服务运行正常 (端口: $port)"
        else
            log_warning "$name 服务可能未启动或不可访问 (端口: $port)"
        fi
    done
}

# 导入Grafana仪表板
import_grafana_dashboards() {
    log_info "等待Grafana启动..."
    sleep 30
    
    log_info "导入Grafana仪表板..."
    
    local dashboards=(
        "monitoring/grafana/dashboards/system-overview.json"
        "monitoring/grafana/dashboards/application-performance.json"
        "monitoring/grafana/dashboards/business-metrics.json"
    )
    
    for dashboard in "${dashboards[@]}"; do
        if [ -f "$dashboard" ]; then
            local dashboard_name=$(basename "$dashboard" .json)
            log_info "导入仪表板: $dashboard_name"
            
            # 这里可以添加通过API导入仪表板的逻辑
            # curl -X POST -H "Content-Type: application/json" \
            #      -d @"$dashboard" \
            #      http://admin:admin123@localhost:3001/api/dashboards/db
        else
            log_warning "仪表板文件不存在: $dashboard"
        fi
    done
    
    log_success "仪表板导入完成"
}

# 显示监控信息
show_monitoring_info() {
    echo ""
    log_success "监控系统设置完成！"
    echo ""
    echo "监控服务访问地址："
    echo "  Prometheus: http://localhost:9090"
    echo "  Grafana: http://localhost:3001 (admin/admin123)"
    echo "  Node Exporter: http://localhost:9100"
    echo "  cAdvisor: http://localhost:8080"
    echo "  AlertManager: http://localhost:9093"
    echo ""
    echo "常用命令："
    echo "  查看监控日志: docker-compose -f docker-compose.monitoring.yml logs -f"
    echo "  停止监控服务: docker-compose -f docker-compose.monitoring.yml down"
    echo "  重启监控服务: docker-compose -f docker-compose.monitoring.yml restart"
    echo ""
}

# 主函数
main() {
    local action=${1:-"setup"}
    
    echo "========================================"
    echo "    监控系统设置脚本"
    echo "========================================"
    echo ""
    
    case "$action" in
        "setup")
            check_dependencies
            create_monitoring_directories
            generate_prometheus_config
            setup_grafana_config
            create_monitoring_compose
            create_alertmanager_config
            start_monitoring_services
            check_monitoring_services
            import_grafana_dashboards
            show_monitoring_info
            ;;
        "start")
            start_monitoring_services
            check_monitoring_services
            ;;
        "stop")
            log_info "停止监控服务..."
            docker-compose -f docker-compose.monitoring.yml down
            log_success "监控服务已停止"
            ;;
        "status")
            check_monitoring_services
            ;;
        "help"|"-h"|"--help")
            echo "用法: $0 [action]"
            echo ""
            echo "操作:"
            echo "  setup   - 完整设置监控系统 (默认)"
            echo "  start   - 启动监控服务"
            echo "  stop    - 停止监控服务"
            echo "  status  - 检查监控服务状态"
            echo "  help    - 显示帮助信息"
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