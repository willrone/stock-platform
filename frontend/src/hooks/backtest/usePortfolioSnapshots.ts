/**
 * 组合快照数据获取 Hook
 */

import { useState, useEffect } from 'react';
import { BacktestService, PortfolioSnapshot } from '@/services/backtestService';

export const usePortfolioSnapshots = (taskId?: string) => {
  const [portfolioSnapshots, setPortfolioSnapshots] = useState<PortfolioSnapshot[]>([]);
  const [loadingSnapshots, setLoadingSnapshots] = useState(false);

  useEffect(() => {
    if (!taskId) {
      return;
    }

    const loadSnapshots = async () => {
      setLoadingSnapshots(true);
      try {
        const result = await BacktestService.getPortfolioSnapshots(
          taskId,
          undefined,
          undefined,
          10000
        );
        if (result && result.snapshots) {
          // 按日期排序
          const sorted = [...result.snapshots].sort(
            (a, b) => new Date(a.snapshot_date).getTime() - new Date(b.snapshot_date).getTime()
          );
          setPortfolioSnapshots(sorted);
        }
      } catch (error) {
        console.error('获取组合快照数据失败:', error);
      } finally {
        setLoadingSnapshots(false);
      }
    };

    loadSnapshots();
  }, [taskId]);

  return { portfolioSnapshots, loadingSnapshots };
};
