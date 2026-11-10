# launchd 定时任务（本项目相关）

这些 plist **不在 LaunchAgents 里跟踪**，这里留一份可复制的版本，
换机器 / 重装时不用重新手写。

## 安装（在**用户自己的终端**执行；Hermes 网关禁止执行 launchctl bootstrap）

```bash
cd ~/Projects/stock-platform/scripts/launchd
for p in com.rone.tdx-update.plist com.rone.paper-portfolio.plist; do
  cp "$p" ~/Library/LaunchAgents/
  launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/"$p" 2>/dev/null || \
    launchctl kickstart -k gui/$(id -u)/$(basename "$p" .plist)
done
launchctl list | grep com.rone
```

## 时点为什么是这样（别随手改）

| 任务 | 时点 | 依据 |
|---|---|---|
| `com.rone.tdx-update` | 工作日 16:05 | TDX 当日日线包 **17:00 后**才发布 → 16:05 跑时数据最新日永远落后 2 天 |
| `com.rone.paper-portfolio` | **周一 17:40** + **周二 09:20** | 纸面盘是**严格 lag1**：信号日 = 最后一个周五，成交日 = 下一个交易日，成交价用成交日收盘价 → 必须等当天的日线包落地（实测 **17:30 左右**）才有成交价。17:00 跑会**竞态失败**（数据没到 → 脚本直接报错退出）。周二 09:20 是补跑（TDX 包偶尔晚到，实测 10-23 的包迟了 2 天）；按**成交日**判重，不会重复下单 |
| Hermes cron `daily-signal-push` | 周一 **18:00** | 必须等纸面盘跑完（3~8 分钟）并写出 `data/paper/portfolio.json` |

## 排错

- 纸面盘日志：`~/Projects/stock-platform/runtime/logs/paper_portfolio_*.log`（保留最近 8 次）
- launchd 日志：`runtime/logs/paper_portfolio_launchd.{out,err}.log`
- **静默失败**是这套东西最大的坑：`SystemExit` 会让 launchd 什么都不显示，
  所以每次改完时点都要**手动跑一次**确认输出。
