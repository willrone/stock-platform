// 数据服务监控面板前端脚本

let refreshInterval;

document.addEventListener('DOMContentLoaded', function() {
    console.log('Parquet数据服务监控面板初始化...');
    refreshAll();

    // 每30秒自动刷新
    refreshInterval = setInterval(refreshAll, 30000);
});

// 页面卸载时清除定时器
window.addEventListener('beforeunload', function() {
    if (refreshInterval) {
        clearInterval(refreshInterval);
    }
});

// 刷新所有数据
async function refreshAll() {
    console.log('正在刷新Parquet数据服务状态...');
    try {
        await Promise.all([
            loadHealthStatus(),
            loadDataSummary()
        ]);
        console.log('Parquet数据服务状态刷新完成');
    } catch (error) {
        console.error('刷新数据失败:', error);
    }
}

// 加载健康状态
async function loadHealthStatus() {
    try {
        console.log('正在加载Parquet服务健康状态...');
        const response = await fetch('/api/data/health');
        const data = await response.json();

        // 存储健康状态数据，稍后在loadDataSummary中一起渲染
        window.healthStatusData = data;
        console.log('Parquet服务健康状态加载完成:', data);

    } catch (error) {
        console.error('加载Parquet服务健康状态失败:', error);
        window.healthStatusData = null;
        showError('加载Parquet服务健康状态失败');
    }
}

// 加载数据汇总
async function loadDataSummary() {
    try {
        console.log('正在加载Parquet数据汇总...');
        const response = await fetch('/api/data/data_summary');
        const data = await response.json();

        if (data.error) {
            console.error('获取Parquet数据汇总失败:', data.error);
            showError('获取Parquet数据汇总失败: ' + data.error);
            return;
        }

        console.log('Parquet数据汇总加载完成:', data);

        const statusGrid = document.getElementById('statusGrid');

        // 清空现有的所有卡片
        statusGrid.innerHTML = '';

        let html = '';

        // 如果有健康状态数据，先添加健康状态卡片
        if (window.healthStatusData) {
            html += `
                <div class="status-card health-card">
                    <h3>🔧 服务健康状态</h3>
                    <div class="metric">
                        <span>状态:</span>
                        <span class="metric-value ${window.healthStatusData.status === 'healthy' ? 'success' : 'danger'}">
                            ${window.healthStatusData.status === 'healthy' ? '✅ 正常' : '❌ 异常'}
                        </span>
                    </div>
                    <div class="metric">
                        <span>Parquet存储:</span>
                        <span class="metric-value ${window.healthStatusData.storage_available ? 'success' : 'danger'}">
                            ${window.healthStatusData.storage_available ? '✅ 可用' : '❌ 不可用'}
                        </span>
                    </div>
                    <div class="metric">
                        <span>存储类型:</span>
                        <span class="metric-value">${window.healthStatusData.storage_type}</span>
                    </div>
                    <div class="metric">
                        <span>消息:</span>
                        <span class="metric-value">${window.healthStatusData.message}</span>
                    </div>
                </div>
            `;
        }

        // 添加Parquet文件统计卡片
        html += `
            <div class="status-card summary-card">
                <h3>📊 Parquet文件统计</h3>
                <div class="metric">
                    <span>总股票数:</span>
                    <span class="metric-value">${data.total_stocks}</span>
                </div>
                <div class="metric">
                    <span>完整数据:</span>
                    <span class="metric-value success">${data.complete_stocks}</span>
                </div>
                <div class="metric">
                    <span>不完整数据:</span>
                    <span class="metric-value warning">${data.incomplete_stocks}</span>
                </div>
                <div class="metric">
                    <span>缺失数据:</span>
                    <span class="metric-value danger">${data.missing_stocks}</span>
                </div>
                <div class="metric">
                    <span>总记录数:</span>
                    <span class="metric-value">${data.total_records ? data.total_records.toLocaleString() : '0'}</span>
                </div>
                <div class="metric">
                    <span>最后更新:</span>
                    <span class="metric-value">${data.last_update || '从未更新'}</span>
                </div>
                ${data.note ? `<div class="metric"><span>备注:</span><span class="metric-value">${data.note}</span></div>` : ''}
            </div>
        `;

        // 一次性设置所有HTML
        statusGrid.innerHTML = html;

    } catch (error) {
        console.error('加载数据汇总失败:', error);
        showError('加载数据汇总失败');
    }
}

// 加载日志
async function loadLogs() {
    const logType = document.getElementById('logType').value;
    const logsContainer = document.getElementById('logsContainer');

    console.log(`正在加载日志: ${logType}`);

    logsContainer.innerHTML = '<div class="loading"></div> 正在加载日志...';

    try {
        const response = await fetch(`/api/data/logs/${logType}`);
        const data = await response.json();

        if (data.error) {
            logsContainer.textContent = `错误: ${data.error}`;
            console.error('加载日志失败:', data.error);
            return;
        }

        logsContainer.textContent = data.content || '日志为空';
        logsContainer.scrollTop = logsContainer.scrollHeight;

        console.log(`日志加载完成: ${data.lines} 行`);

    } catch (error) {
        console.error('加载日志失败:', error);
        logsContainer.textContent = '加载日志失败，请检查网络连接';
    }
}

// 清空日志显示
function clearLogs() {
    document.getElementById('logsContainer').textContent = '点击"加载日志"查看内容...';
    console.log('日志显示已清空');
}

// 手动获取数据到Parquet
async function manualFetch() {
    const btn = document.getElementById('fetchBtn');
    const originalText = btn.textContent;

    console.log('开始手动获取数据到Parquet...');

    btn.disabled = true;
    btn.textContent = '⏳ 获取中...';

    try {
        const response = await fetch('/api/data/manual_fetch', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        const data = await response.json();

        if (data.error) {
            console.error('手动获取失败:', data.error);
            showError('手动获取失败: ' + data.error);
        } else {
            console.log('手动获取成功:', data.message);
            showSuccess(data.message);
            // 3秒后重新加载日志
            setTimeout(() => loadLogs(), 3000);
        }

    } catch (error) {
        console.error('手动获取失败:', error);
        showError('手动获取失败，请检查网络连接');
    } finally {
        btn.disabled = false;
        btn.textContent = originalText;
    }
}

// 手动同步功能（已废弃，Parquet存储无需同步）
// 保留函数以防调用，但不执行实际操作
async function manualSync() {
    showSuccess('Parquet存储无需同步操作');
}

// 显示成功消息
function showSuccess(message) {
    // 简单的成功提示，可以扩展为更复杂的UI
    alert('✅ ' + message);
}

// 显示错误消息
function showError(message) {
    alert('❌ ' + message);
}
