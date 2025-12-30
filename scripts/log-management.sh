#!/bin/bash

# 日志管理脚本
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
LOG_DIR="./backend/data/logs"
ARCHIVE_DIR="./backend/data/logs/archive"
MAX_LOG_SIZE="100M"
RETENTION_DAYS=30
COMPRESSION_DAYS=7

# 创建归档目录
create_archive_dir() {
    if [ ! -d "$ARCHIVE_DIR" ]; then
        mkdir -p "$ARCHIVE_DIR"
        log_info "创建归档目录: $ARCHIVE_DIR"
    fi
}

# 压缩旧日志文件
compress_old_logs() {
    log_info "压缩 $COMPRESSION_DAYS 天前的日志文件..."
    
    find "$LOG_DIR" -name "*.log" -type f -mtime +$COMPRESSION_DAYS ! -path "*/archive/*" | while read -r file; do
        if [ -f "$file" ]; then
            log_info "压缩文件: $file"
            gzip "$file"
            log_success "已压缩: $file.gz"
        fi
    done
}

# 归档旧日志文件
archive_old_logs() {
    log_info "归档 $RETENTION_DAYS 天前的日志文件..."
    
    find "$LOG_DIR" -name "*.gz" -type f -mtime +$RETENTION_DAYS ! -path "*/archive/*" | while read -r file; do
        if [ -f "$file" ]; then
            log_info "归档文件: $file"
            mv "$file" "$ARCHIVE_DIR/"
            log_success "已归档: $(basename $file)"
        fi
    done
}

# 清理过期的归档文件
cleanup_archived_logs() {
    local archive_retention_days=$((RETENTION_DAYS * 3))  # 归档文件保留3倍时间
    log_info "清理 $archive_retention_days 天前的归档文件..."
    
    find "$ARCHIVE_DIR" -name "*.gz" -type f -mtime +$archive_retention_days | while read -r file; do
        if [ -f "$file" ]; then
            log_info "删除过期归档: $file"
            rm "$file"
            log_success "已删除: $(basename $file)"
        fi
    done
}

# 检查日志文件大小
check_log_sizes() {
    log_info "检查日志文件大小..."
    
    find "$LOG_DIR" -name "*.log" -type f ! -path "*/archive/*" | while read -r file; do
        if [ -f "$file" ]; then
            size=$(du -h "$file" | cut -f1)
            log_info "$(basename $file): $size"
            
            # 检查是否超过最大大小
            size_bytes=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo 0)
            max_size_bytes=$(echo "$MAX_LOG_SIZE" | sed 's/M/*1024*1024/' | bc)
            
            if [ "$size_bytes" -gt "$max_size_bytes" ]; then
                log_warning "文件 $(basename $file) 大小 ($size) 超过限制 ($MAX_LOG_SIZE)"
            fi
        fi
    done
}

# 生成日志统计报告
generate_log_report() {
    log_info "生成日志统计报告..."
    
    local report_file="$LOG_DIR/log_report_$(date +%Y%m%d).txt"
    
    {
        echo "=========================================="
        echo "日志统计报告 - $(date)"
        echo "=========================================="
        echo ""
        
        echo "日志目录: $LOG_DIR"
        echo "归档目录: $ARCHIVE_DIR"
        echo "保留天数: $RETENTION_DAYS"
        echo "压缩天数: $COMPRESSION_DAYS"
        echo ""
        
        echo "当前日志文件:"
        find "$LOG_DIR" -name "*.log" -type f ! -path "*/archive/*" -exec ls -lh {} \; | awk '{print $9, $5, $6, $7, $8}'
        echo ""
        
        echo "压缩日志文件:"
        find "$LOG_DIR" -name "*.gz" -type f ! -path "*/archive/*" -exec ls -lh {} \; | awk '{print $9, $5, $6, $7, $8}' | head -10
        echo ""
        
        echo "归档文件统计:"
        if [ -d "$ARCHIVE_DIR" ]; then
            archive_count=$(find "$ARCHIVE_DIR" -name "*.gz" -type f | wc -l)
            archive_size=$(du -sh "$ARCHIVE_DIR" 2>/dev/null | cut -f1 || echo "0")
            echo "归档文件数量: $archive_count"
            echo "归档总大小: $archive_size"
        else
            echo "无归档文件"
        fi
        echo ""
        
        echo "磁盘使用情况:"
        df -h "$LOG_DIR" | tail -1
        echo ""
        
        echo "最近的错误日志 (最后10行):"
        if [ -f "$LOG_DIR/error.log" ]; then
            tail -10 "$LOG_DIR/error.log" | sed 's/^/  /'
        else
            echo "  无错误日志文件"
        fi
        
    } > "$report_file"
    
    log_success "报告已生成: $report_file"
    
    # 显示报告摘要
    echo ""
    log_info "报告摘要:"
    head -20 "$report_file" | tail -15
}

# 清理临时日志文件
cleanup_temp_logs() {
    log_info "清理临时日志文件..."
    
    # 清理空日志文件
    find "$LOG_DIR" -name "*.log" -type f -empty ! -path "*/archive/*" | while read -r file; do
        if [ -f "$file" ]; then
            log_info "删除空日志文件: $file"
            rm "$file"
        fi
    done
    
    # 清理临时文件
    find "$LOG_DIR" -name "*.tmp" -type f | while read -r file; do
        if [ -f "$file" ]; then
            log_info "删除临时文件: $file"
            rm "$file"
        fi
    done
}

# 监控日志错误
monitor_errors() {
    log_info "监控最近的错误..."
    
    local error_file="$LOG_DIR/error.log"
    if [ -f "$error_file" ]; then
        local recent_errors=$(tail -100 "$error_file" | grep -c "ERROR" || echo 0)
        local critical_errors=$(tail -100 "$error_file" | grep -c "CRITICAL" || echo 0)
        
        log_info "最近100行中的错误数量: $recent_errors"
        log_info "最近100行中的严重错误数量: $critical_errors"
        
        if [ "$critical_errors" -gt 0 ]; then
            log_error "发现严重错误，请检查日志文件"
            tail -20 "$error_file" | grep "CRITICAL" | tail -5
        fi
    else
        log_info "未找到错误日志文件"
    fi
}

# 主函数
main() {
    local action=${1:-"all"}
    
    echo "========================================"
    echo "    日志管理脚本"
    echo "========================================"
    echo ""
    
    if [ ! -d "$LOG_DIR" ]; then
        log_error "日志目录不存在: $LOG_DIR"
        exit 1
    fi
    
    case "$action" in
        "compress")
            create_archive_dir
            compress_old_logs
            ;;
        "archive")
            create_archive_dir
            archive_old_logs
            ;;
        "cleanup")
            create_archive_dir
            cleanup_archived_logs
            cleanup_temp_logs
            ;;
        "check")
            check_log_sizes
            monitor_errors
            ;;
        "report")
            generate_log_report
            ;;
        "all")
            create_archive_dir
            compress_old_logs
            archive_old_logs
            cleanup_archived_logs
            cleanup_temp_logs
            check_log_sizes
            monitor_errors
            generate_log_report
            ;;
        "help"|"-h"|"--help")
            echo "用法: $0 [action]"
            echo ""
            echo "操作:"
            echo "  compress  - 压缩旧日志文件"
            echo "  archive   - 归档旧日志文件"
            echo "  cleanup   - 清理过期文件"
            echo "  check     - 检查日志状态"
            echo "  report    - 生成统计报告"
            echo "  all       - 执行所有操作 (默认)"
            echo "  help      - 显示帮助信息"
            ;;
        *)
            log_error "未知操作: $action"
            echo "使用 '$0 help' 查看帮助信息"
            exit 1
            ;;
    esac
    
    echo ""
    log_success "日志管理操作完成"
}

# 执行主函数
main "$@"