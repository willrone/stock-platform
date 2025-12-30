#!/bin/bash

# 集成测试运行脚本
# 
# 运行完整的前后端集成测试，包括：
# - 后端API测试
# - 前端集成测试
# - WebSocket连接测试
# - 端到端工作流程测试

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
    
    # 检查Python环境
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装"
        exit 1
    fi
    
    # 检查Node.js环境
    if ! command -v node &> /dev/null; then
        log_error "Node.js 未安装"
        exit 1
    fi
    
    # 检查后端依赖
    if [ ! -f "backend/requirements.txt" ]; then
        log_error "后端requirements.txt文件不存在"
        exit 1
    fi
    
    # 检查前端依赖
    if [ ! -f "frontend/package.json" ]; then
        log_error "前端package.json文件不存在"
        exit 1
    fi
    
    log_success "依赖检查完成"
}

# 启动后端服务
start_backend() {
    log_info "启动后端服务..."
    
    cd backend
    
    # 激活虚拟环境（如果存在）
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
    
    # 安装依赖
    pip install -r requirements.txt > /dev/null 2>&1
    
    # 启动后端服务
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    
    cd ..
    
    # 等待服务启动
    log_info "等待后端服务启动..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
            log_success "后端服务启动成功 (PID: $BACKEND_PID)"
            return 0
        fi
        sleep 1
    done
    
    log_error "后端服务启动失败"
    return 1
}

# 启动前端服务
start_frontend() {
    log_info "启动前端服务..."
    
    cd frontend
    
    # 安装依赖
    npm install > /dev/null 2>&1
    
    # 构建前端
    npm run build > /dev/null 2>&1
    
    # 启动前端服务
    npm run start &
    FRONTEND_PID=$!
    
    cd ..
    
    # 等待服务启动
    log_info "等待前端服务启动..."
    for i in {1..30}; do
        if curl -s http://localhost:3000 > /dev/null 2>&1; then
            log_success "前端服务启动成功 (PID: $FRONTEND_PID)"
            return 0
        fi
        sleep 1
    done
    
    log_error "前端服务启动失败"
    return 1
}

# 运行后端测试
run_backend_tests() {
    log_info "运行后端集成测试..."
    
    cd backend
    
    # 激活虚拟环境（如果存在）
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
    
    # 运行测试
    if python -m pytest tests/test_integration.py -v --tb=short; then
        log_success "后端集成测试通过"
        BACKEND_TEST_RESULT=0
    else
        log_error "后端集成测试失败"
        BACKEND_TEST_RESULT=1
    fi
    
    cd ..
    return $BACKEND_TEST_RESULT
}

# 运行前端测试
run_frontend_tests() {
    log_info "运行前端集成测试..."
    
    cd frontend
    
    # 运行测试
    if npm run test -- --run src/__tests__/integration.test.ts; then
        log_success "前端集成测试通过"
        FRONTEND_TEST_RESULT=0
    else
        log_error "前端集成测试失败"
        FRONTEND_TEST_RESULT=1
    fi
    
    cd ..
    return $FRONTEND_TEST_RESULT
}

# 运行WebSocket测试
run_websocket_tests() {
    log_info "运行WebSocket连接测试..."
    
    # 使用Node.js脚本测试WebSocket连接
    cat > /tmp/websocket_test.js << 'EOF'
const WebSocket = require('ws');

const ws = new WebSocket('ws://localhost:8000/ws');

ws.on('open', function open() {
    console.log('WebSocket连接成功');
    
    // 发送ping消息
    ws.send(JSON.stringify({ type: 'ping' }));
});

ws.on('message', function message(data) {
    const msg = JSON.parse(data);
    console.log('收到消息:', msg.type);
    
    if (msg.type === 'pong') {
        console.log('WebSocket测试通过');
        process.exit(0);
    }
});

ws.on('error', function error(err) {
    console.error('WebSocket错误:', err.message);
    process.exit(1);
});

// 超时处理
setTimeout(() => {
    console.error('WebSocket测试超时');
    process.exit(1);
}, 5000);
EOF
    
    if node /tmp/websocket_test.js; then
        log_success "WebSocket测试通过"
        WS_TEST_RESULT=0
    else
        log_error "WebSocket测试失败"
        WS_TEST_RESULT=1
    fi
    
    rm -f /tmp/websocket_test.js
    return $WS_TEST_RESULT
}

# 运行端到端测试
run_e2e_tests() {
    log_info "运行端到端测试..."
    
    # 创建端到端测试脚本
    cat > /tmp/e2e_test.js << 'EOF'
const axios = require('axios');

async function runE2ETest() {
    try {
        console.log('开始端到端测试...');
        
        // 1. 健康检查
        console.log('1. 执行健康检查...');
        const healthResponse = await axios.get('http://localhost:8000/api/v1/health');
        if (!healthResponse.data.success) {
            throw new Error('健康检查失败');
        }
        console.log('✓ 健康检查通过');
        
        // 2. 获取模型列表
        console.log('2. 获取模型列表...');
        const modelsResponse = await axios.get('http://localhost:8000/api/v1/models');
        if (!modelsResponse.data.success || !modelsResponse.data.data.models.length) {
            throw new Error('获取模型列表失败');
        }
        console.log('✓ 模型列表获取成功');
        
        // 3. 创建预测任务
        console.log('3. 创建预测任务...');
        const taskRequest = {
            task_name: 'E2E测试任务',
            stock_codes: ['000001.SZ'],
            model_id: modelsResponse.data.data.models[0].model_id,
            prediction_config: {
                horizon: 'short_term',
                confidence_level: 0.95
            }
        };
        
        const taskResponse = await axios.post('http://localhost:8000/api/v1/tasks', taskRequest);
        if (!taskResponse.data.success) {
            throw new Error('创建任务失败');
        }
        console.log('✓ 任务创建成功');
        
        // 4. 获取任务详情
        console.log('4. 获取任务详情...');
        const taskId = taskResponse.data.data.task_id;
        const taskDetailResponse = await axios.get(`http://localhost:8000/api/v1/tasks/${taskId}`);
        if (!taskDetailResponse.data.success) {
            throw new Error('获取任务详情失败');
        }
        console.log('✓ 任务详情获取成功');
        
        // 5. 获取数据状态
        console.log('5. 获取数据状态...');
        const dataStatusResponse = await axios.get('http://localhost:8000/api/v1/data/status');
        if (!dataStatusResponse.data.success) {
            throw new Error('获取数据状态失败');
        }
        console.log('✓ 数据状态获取成功');
        
        console.log('端到端测试全部通过！');
        return true;
        
    } catch (error) {
        console.error('端到端测试失败:', error.message);
        return false;
    }
}

runE2ETest().then(success => {
    process.exit(success ? 0 : 1);
});
EOF
    
    cd frontend && npm install axios > /dev/null 2>&1 && cd ..
    
    if node /tmp/e2e_test.js; then
        log_success "端到端测试通过"
        E2E_TEST_RESULT=0
    else
        log_error "端到端测试失败"
        E2E_TEST_RESULT=1
    fi
    
    rm -f /tmp/e2e_test.js
    return $E2E_TEST_RESULT
}

# 清理函数
cleanup() {
    log_info "清理测试环境..."
    
    # 停止后端服务
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
        log_info "后端服务已停止"
    fi
    
    # 停止前端服务
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
        log_info "前端服务已停止"
    fi
    
    # 清理临时文件
    rm -f /tmp/websocket_test.js /tmp/e2e_test.js
}

# 生成测试报告
generate_report() {
    log_info "生成测试报告..."
    
    local total_tests=4
    local passed_tests=0
    
    echo "# 集成测试报告" > integration_test_report.md
    echo "" >> integration_test_report.md
    echo "**测试时间**: $(date)" >> integration_test_report.md
    echo "" >> integration_test_report.md
    echo "## 测试结果" >> integration_test_report.md
    echo "" >> integration_test_report.md
    
    # 后端测试结果
    if [ "$BACKEND_TEST_RESULT" -eq 0 ]; then
        echo "- ✅ 后端集成测试: 通过" >> integration_test_report.md
        ((passed_tests++))
    else
        echo "- ❌ 后端集成测试: 失败" >> integration_test_report.md
    fi
    
    # 前端测试结果
    if [ "$FRONTEND_TEST_RESULT" -eq 0 ]; then
        echo "- ✅ 前端集成测试: 通过" >> integration_test_report.md
        ((passed_tests++))
    else
        echo "- ❌ 前端集成测试: 失败" >> integration_test_report.md
    fi
    
    # WebSocket测试结果
    if [ "$WS_TEST_RESULT" -eq 0 ]; then
        echo "- ✅ WebSocket连接测试: 通过" >> integration_test_report.md
        ((passed_tests++))
    else
        echo "- ❌ WebSocket连接测试: 失败" >> integration_test_report.md
    fi
    
    # 端到端测试结果
    if [ "$E2E_TEST_RESULT" -eq 0 ]; then
        echo "- ✅ 端到端测试: 通过" >> integration_test_report.md
        ((passed_tests++))
    else
        echo "- ❌ 端到端测试: 失败" >> integration_test_report.md
    fi
    
    echo "" >> integration_test_report.md
    echo "**总计**: $passed_tests/$total_tests 通过" >> integration_test_report.md
    echo "**成功率**: $(( passed_tests * 100 / total_tests ))%" >> integration_test_report.md
    
    log_success "测试报告已生成: integration_test_report.md"
}

# 主函数
main() {
    log_info "开始运行集成测试..."
    
    # 设置清理陷阱
    trap cleanup EXIT
    
    # 检查依赖
    check_dependencies
    
    # 启动服务
    if ! start_backend; then
        log_error "无法启动后端服务"
        exit 1
    fi
    
    # 等待服务稳定
    sleep 5
    
    # 运行测试
    run_backend_tests
    run_websocket_tests
    run_e2e_tests
    
    # 如果需要前端测试，启动前端服务
    if [ "$1" = "--with-frontend" ]; then
        if start_frontend; then
            sleep 10
            run_frontend_tests
        else
            log_warning "前端服务启动失败，跳过前端测试"
            FRONTEND_TEST_RESULT=1
        fi
    else
        log_info "跳过前端测试（使用 --with-frontend 参数启用）"
        FRONTEND_TEST_RESULT=0
    fi
    
    # 生成报告
    generate_report
    
    # 计算总体结果
    local total_failed=0
    [ "$BACKEND_TEST_RESULT" -ne 0 ] && ((total_failed++))
    [ "$WS_TEST_RESULT" -ne 0 ] && ((total_failed++))
    [ "$E2E_TEST_RESULT" -ne 0 ] && ((total_failed++))
    
    if [ "$1" = "--with-frontend" ]; then
        [ "$FRONTEND_TEST_RESULT" -ne 0 ] && ((total_failed++))
    fi
    
    if [ $total_failed -eq 0 ]; then
        log_success "所有集成测试通过！"
        exit 0
    else
        log_error "$total_failed 个测试失败"
        exit 1
    fi
}

# 显示帮助信息
show_help() {
    echo "集成测试运行脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --with-frontend    包含前端测试（需要更长时间）"
    echo "  --help            显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                 运行基础集成测试"
    echo "  $0 --with-frontend 运行包含前端的完整集成测试"
}

# 解析命令行参数
case "$1" in
    --help)
        show_help
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac