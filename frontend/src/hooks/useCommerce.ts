import { useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { message } from 'antd';
import { EVENT_TYPES } from '@/services/api/commerce';
import { commerceService } from '@/services/commerce/commerce.service';

/**
 * 自定义 Hook：用于在组件中进行计费检查和记录
 * 
 * 使用方式：
 * const { canProceed, showCost, recordUsage, balance } = useCommerce();
 * 
 * if (canProceed) {
 *   // 执行业务逻辑
 *   await recordUsage(EVENT_TYPES.BACKTEST_BASIC);
 * }
 */
export function useCommerce() {
  const router = useRouter();
  const [balance, setBalance] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [checkResult, setCheckResult] = useState<{ can_proceed: boolean; message: string; estimated_cost_yuan: number } | null>(null);

  // 初始化：查询余额
  const initialize = useCallback(async () => {
    try {
      setLoading(true);
      const balanceInfo = await commerceService.getBalance();
      setBalance(balanceInfo.balance_yuan);
    } catch (e) {
      message.error('获取余额失败');
    } finally {
      setLoading(false);
    }
  }, []);

  // 检查是否可以执行指定操作
  const checkCanExecute = useCallback(async (event_type: string) => {
    try {
      const result = await commerceService.checkCanExecute(event_type);
      setCheckResult(result);
      return result;
    } catch (e) {
      message.error('检查失败');
      return { can_proceed: false, message: '检查失败，请重试', estimated_cost_yuan: 0 };
    }
  }, []);

  // 记录用量并扣费
  const recordUsage = useCallback(async (
    event_type: string, 
    quantity = 1, 
    metadata?: Record<string, any>
  ) => {
    try {
      // 先检查权限
      const check = await checkCanExecute(event_type);
      if (!check.can_proceed) {
        throw new Error(check.message);
      }

      // 记录用量
      const result = await commerceService.recordUsage(event_type, quantity, metadata);
      
      // 更新余额（扣除费用）
      if (balance !== null) {
        setBalance(balance - result.total_cost);
      }

      message.success(`本次操作花费 ¥${result.total_cost.toFixed(2)}`);
      return result;
    } catch (e: any) {
      message.error(e.message || '操作失败');
      throw e;
    }
  }, [balance, checkCanExecute]);

  // 充值
  const deposit = useCallback(async (amount: number) => {
    try {
      await commerceService.deposit(amount);
      // 重新查询余额
      const balanceInfo = await commerceService.getBalance();
      setBalance(balanceInfo.balance_yuan);
      message.success(`充值成功！当前余额 ¥${balanceInfo.balance_yuan.toFixed(2)}`);
    } catch (e) {
      message.error('充值失败');
    }
  }, []);

  // 刷新余额
  const refreshBalance = useCallback(async () => {
    try {
      const balanceInfo = await commerceService.getBalance();
      setBalance(balanceInfo.balance_yuan);
    } catch (e) {
      message.error('刷新余额失败');
    }
  }, []);

  // 自动初始化
  // 注意：这里使用 useEffect 会导致在服务器端执行，在客户端生效
  // 实际项目中应该在组件内部调用 initialize()
  
  return {
    // 状态
    balance,
    loading,
    checkResult,
    
    // 方法
    initialize,
    checkCanExecute,
    recordUsage,
    deposit,
    refreshBalance,
  };
}