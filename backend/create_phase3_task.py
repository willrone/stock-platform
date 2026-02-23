"""
创建 Phase 3 优化验证回测任务
"""

import requests
import json
import time

# 基线任务配置
config = {
    "stock_codes": [
        "000001.SZ", "000002.SZ", "000004.SZ", "000006.SZ", "000007.SZ", "000008.SZ", "000009.SZ", "000010.SZ", "000011.SZ", "000012.SZ",
        "000014.SZ", "000016.SZ", "000017.SZ", "000019.SZ", "000020.SZ", "000021.SZ", "001872.SZ", "000025.SZ", "000026.SZ", "000027.SZ",
        "000028.SZ", "000029.SZ", "000030.SZ", "000031.SZ", "000032.SZ", "000034.SZ", "000035.SZ", "000036.SZ", "000037.SZ", "000039.SZ",
        "000042.SZ", "001914.SZ", "000045.SZ", "000048.SZ", "000049.SZ", "000050.SZ", "000055.SZ", "000056.SZ", "000058.SZ", "000059.SZ",
        "000060.SZ", "000061.SZ", "000062.SZ", "000063.SZ", "000065.SZ", "000066.SZ", "000068.SZ", "000069.SZ", "000070.SZ", "000078.SZ",
        "000088.SZ", "000089.SZ", "000090.SZ", "000096.SZ", "000099.SZ", "000100.SZ", "000151.SZ", "000153.SZ", "000155.SZ", "000156.SZ",
        "000157.SZ", "000158.SZ", "000159.SZ", "000301.SZ", "000400.SZ", "000401.SZ", "000402.SZ", "000403.SZ", "000404.SZ", "000407.SZ",
        "000408.SZ", "000409.SZ", "000410.SZ", "000411.SZ", "000415.SZ", "000417.SZ", "000419.SZ", "000420.SZ", "000421.SZ", "000422.SZ",
        "000423.SZ", "000425.SZ", "000426.SZ", "000428.SZ", "000429.SZ", "000430.SZ", "000488.SZ", "000498.SZ", "000501.SZ", "000503.SZ",
        "000504.SZ", "000505.SZ", "000506.SZ", "000507.SZ", "000509.SZ", "000510.SZ", "000513.SZ", "000514.SZ", "000516.SZ", "000517.SZ",
        "000518.SZ", "000519.SZ", "000520.SZ", "000521.SZ", "000523.SZ", "000524.SZ", "000525.SZ", "000526.SZ", "000528.SZ", "000529.SZ",
        "000530.SZ", "000531.SZ", "000532.SZ", "000533.SZ", "000534.SZ", "000536.SZ", "000537.SZ", "000538.SZ", "000539.SZ", "000541.SZ",
        "000543.SZ", "000544.SZ", "000545.SZ", "000546.SZ", "000547.SZ", "000548.SZ", "000550.SZ", "000551.SZ", "000552.SZ", "000553.SZ",
        "000554.SZ", "000555.SZ", "000557.SZ", "000558.SZ", "000559.SZ", "000560.SZ", "000561.SZ", "000563.SZ", "000564.SZ", "000565.SZ",
        "000566.SZ", "000567.SZ", "000568.SZ", "000570.SZ", "000571.SZ", "000572.SZ", "000573.SZ", "000576.SZ", "000581.SZ", "000582.SZ",
        "000586.SZ", "000589.SZ", "000590.SZ", "000591.SZ", "000592.SZ", "000593.SZ", "000595.SZ", "000596.SZ", "000597.SZ", "000598.SZ",
        "000599.SZ", "000600.SZ", "000601.SZ", "000603.SZ", "000605.SZ", "000607.SZ", "000608.SZ", "000609.SZ", "000610.SZ", "000612.SZ",
        "000615.SZ", "000617.SZ", "000619.SZ", "000620.SZ", "000623.SZ", "000625.SZ", "000626.SZ", "000628.SZ", "000629.SZ", "000630.SZ",
        "000631.SZ", "000632.SZ", "000633.SZ", "000635.SZ", "000636.SZ", "000637.SZ", "000638.SZ", "000639.SZ", "000650.SZ", "000651.SZ",
        "000652.SZ", "000655.SZ", "000656.SZ", "000657.SZ", "000659.SZ", "000661.SZ", "000663.SZ", "000665.SZ", "000668.SZ", "000669.SZ",
        "000670.SZ", "000672.SZ", "000676.SZ", "000677.SZ", "000678.SZ", "000679.SZ", "000680.SZ", "000681.SZ", "000682.SZ", "000683.SZ",
        "000685.SZ", "000686.SZ", "000688.SZ", "000690.SZ", "000691.SZ", "000692.SZ", "000695.SZ", "001696.SZ", "000697.SZ", "000698.SZ",
        "000700.SZ", "000701.SZ", "000702.SZ", "000703.SZ", "000705.SZ", "000707.SZ", "000708.SZ", "000709.SZ", "000710.SZ", "000711.SZ",
        "000712.SZ", "000713.SZ", "000715.SZ", "000716.SZ", "000717.SZ", "000718.SZ", "000719.SZ", "000720.SZ", "000721.SZ", "000722.SZ",
        "000723.SZ", "000725.SZ", "000726.SZ", "000727.SZ", "000728.SZ", "000729.SZ", "000731.SZ", "000733.SZ", "000735.SZ", "000736.SZ",
        "000737.SZ", "000738.SZ", "000739.SZ", "000750.SZ", "000751.SZ", "000752.SZ", "000753.SZ", "000755.SZ", "000756.SZ", "000757.SZ",
        "000758.SZ", "000759.SZ", "000761.SZ", "000762.SZ", "000766.SZ", "000767.SZ", "000768.SZ", "000776.SZ", "000777.SZ", "000778.SZ",
        "000779.SZ", "000782.SZ", "000783.SZ", "000785.SZ", "000786.SZ", "000788.SZ", "000789.SZ", "000790.SZ", "000791.SZ", "000792.SZ",
        "000793.SZ", "000795.SZ", "000796.SZ", "000797.SZ", "000798.SZ", "000799.SZ", "000800.SZ", "000801.SZ", "000802.SZ", "000803.SZ",
        "000807.SZ", "000809.SZ", "000810.SZ", "000811.SZ", "000812.SZ", "000813.SZ", "000815.SZ", "000816.SZ", "000818.SZ", "000819.SZ",
        "000820.SZ", "000821.SZ", "000822.SZ", "000823.SZ", "000825.SZ", "000826.SZ", "000828.SZ", "000829.SZ", "000830.SZ", "000831.SZ",
        "000833.SZ", "000837.SZ", "000838.SZ", "000839.SZ", "000848.SZ", "000850.SZ", "000852.SZ", "000856.SZ", "000858.SZ", "000859.SZ",
        "000860.SZ", "000862.SZ", "000863.SZ", "000868.SZ", "000869.SZ", "000875.SZ", "000876.SZ", "000877.SZ", "000878.SZ", "000880.SZ",
        "000881.SZ", "000882.SZ", "000883.SZ", "000885.SZ", "000886.SZ", "000887.SZ", "000888.SZ", "000889.SZ", "000890.SZ", "000892.SZ",
        "000893.SZ", "000895.SZ", "001896.SZ", "000897.SZ", "000898.SZ", "000899.SZ", "000900.SZ", "000901.SZ", "000902.SZ", "000903.SZ",
        "000905.SZ", "000906.SZ", "000908.SZ", "000909.SZ", "000910.SZ", "000911.SZ", "000912.SZ", "000913.SZ", "000915.SZ", "000917.SZ",
        "000919.SZ", "000920.SZ", "000921.SZ", "000922.SZ", "000923.SZ", "000925.SZ", "000926.SZ", "000927.SZ", "000928.SZ", "000929.SZ",
        "000930.SZ", "000931.SZ", "000932.SZ", "000933.SZ", "000935.SZ", "000936.SZ", "000937.SZ", "000938.SZ", "000948.SZ", "000949.SZ",
        "000950.SZ", "000951.SZ", "000952.SZ", "000953.SZ", "000955.SZ", "000957.SZ", "000958.SZ", "000959.SZ", "000960.SZ", "000962.SZ",
        "000963.SZ", "000965.SZ", "000966.SZ", "000967.SZ", "000968.SZ", "000969.SZ", "000970.SZ", "000972.SZ", "000973.SZ", "000975.SZ",
        "000977.SZ", "000978.SZ", "000980.SZ", "000981.SZ", "000983.SZ", "000985.SZ", "000987.SZ", "000988.SZ", "000989.SZ", "000990.SZ",
        "000993.SZ", "000995.SZ", "000997.SZ", "000998.SZ", "000999.SZ", "002001.SZ", "002181.SZ", "600000.SH", "600004.SH", "600006.SH",
        "600007.SH", "600008.SH", "600009.SH", "600010.SH", "600011.SH", "600012.SH", "600015.SH", "600016.SH", "600019.SH", "600020.SH",
        "600021.SH", "600026.SH", "600028.SH", "600029.SH", "600030.SH", "600031.SH", "600033.SH", "600035.SH", "600036.SH", "600037.SH",
        "600038.SH", "600039.SH", "600050.SH", "600051.SH", "600052.SH", "600053.SH", "600054.SH", "600055.SH", "600056.SH", "600057.SH",
        "600058.SH", "600059.SH", "600060.SH", "600061.SH", "600062.SH", "600063.SH", "600064.SH", "600066.SH", "600067.SH", "600071.SH",
        "600072.SH", "600073.SH", "600075.SH", "600076.SH", "600078.SH", "600079.SH", "600080.SH", "600081.SH", "600082.SH", "600084.SH",
        "600085.SH", "600088.SH", "600089.SH", "600094.SH", "600095.SH", "600096.SH", "600097.SH", "600098.SH", "600099.SH", "600100.SH",
        "600101.SH", "600103.SH", "600104.SH", "600105.SH", "600106.SH", "600107.SH", "600108.SH", "600109.SH", "600110.SH", "600111.SH",
        "600113.SH", "600114.SH", "600115.SH", "600116.SH", "600117.SH", "600118.SH", "600119.SH", "600120.SH", "600121.SH", "600123.SH"
    ],
    "start_date": "2023-02-04",
    "end_date": "2026-02-04",
    "initial_cash": 1000000,
    "commission_rate": 0.0003,
    "strategy_name": "RSI",
    "strategy_config": {
        "rsi_period": 14,
        "oversold": 30,
        "overbought": 70
    },
    "enable_performance_profiling": True
}

print("=" * 60)
print("Phase 3 优化验证 - 创建回测任务")
print("=" * 60)
print(f"股票数量: {len(config['stock_codes'])}")
print(f"日期范围: {config['start_date']} ~ {config['end_date']}")
print(f"策略: {config['strategy_name']}")
print(f"参数: {config['strategy_config']}")
print()

# 创建任务
url = "http://localhost:8000/api/v1/tasks"
payload = {
    "task_name": "Phase3优化验证-500股票3年RSI",
    "task_type": "backtest",
    "stock_codes": config["stock_codes"],
    "backtest_config": {
        "start_date": config["start_date"],
        "end_date": config["end_date"],
        "initial_cash": config["initial_cash"],
        "commission_rate": config["commission_rate"],
        "strategy_name": config["strategy_name"],
        "strategy_config": config["strategy_config"],
        "enable_performance_profiling": config["enable_performance_profiling"]
    }
}

print("正在创建任务...")
response = requests.post(url, json=payload)

if response.status_code == 200:
    result = response.json()
    task_id = result["data"]["task_id"]
    print(f"✅ 任务创建成功！")
    print(f"Task ID: {task_id}")
    print()
    
    # 保存 task_id 到文件
    with open("/Users/ronghui/Projects/willrone/backend/phase3_task_id.txt", "w") as f:
        f.write(task_id)
    
    print("=" * 60)
    print("监控任务进度...")
    print("=" * 60)
    
    start_time = time.time()
    last_progress = -1
    
    while True:
        # 查询任务状态
        status_url = f"http://localhost:8000/api/v1/tasks/{task_id}"
        status_response = requests.get(status_url)
        
        if status_response.status_code == 200:
            task_data = status_response.json()["data"]
            status = task_data["status"]
            progress = task_data.get("progress", 0)
            
            # 只在进度变化时打印
            if progress != last_progress:
                elapsed = time.time() - start_time
                print(f"[{elapsed:.1f}s] 状态: {status}, 进度: {progress}%")
                last_progress = progress
            
            if status == "completed":
                elapsed = time.time() - start_time
                print()
                print("=" * 60)
                print("✅ 任务完成！")
                print("=" * 60)
                print(f"总耗时: {elapsed:.2f} 秒")
                
                # 获取结果
                result = task_data.get("result", {})
                if result:
                    print()
                    print("回测结果:")
                    print(f"  总收益率: {result.get('total_return_pct', 'N/A')}%")
                    print(f"  夏普比率: {result.get('sharpe_ratio', 'N/A')}")
                    print(f"  最大回撤: {result.get('max_drawdown_pct', 'N/A')}%")
                    print(f"  总交易: {result.get('total_trades', 'N/A')}")
                    
                    # 性能数据
                    perf = result.get('perf_breakdown', {})
                    if perf:
                        print()
                        print("性能分解:")
                        print(f"  数据加载: {perf.get('data_loading_s', 0):.2f}s")
                        print(f"  信号预计算: {perf.get('precompute_signals_s', 0):.2f}s")
                        print(f"  数组对齐: {perf.get('align_arrays_s', 0):.2f}s")
                        print(f"  主循环: {perf.get('main_loop_s', 0):.2f}s")
                        print(f"  指标计算: {perf.get('metrics_s', 0):.2f}s")
                        print(f"  报告生成: {perf.get('report_generation_s', 0):.2f}s")
                        print(f"  总耗时: {perf.get('total_wall_s', 0):.2f}s")
                
                break
            
            elif status == "failed":
                print()
                print("❌ 任务失败！")
                error = task_data.get("error_message", "未知错误")
                print(f"错误信息: {error}")
                break
        
        time.sleep(2)
    
else:
    print(f"❌ 任务创建失败: {response.status_code}")
    print(response.text)
