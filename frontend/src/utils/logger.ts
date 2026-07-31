// 前端日志工具 - 生产环境下可关闭的 console 封装

const IS_PROD = process.env.NODE_ENV === 'production';

export const logger = {
  debug: (...args: unknown[]) => {
    if (!IS_PROD) console.debug('[DEBUG]', ...args);
  },
  info: (...args: unknown[]) => {
    if (!IS_PROD) console.info('[INFO]', ...args);
  },
  warn: (...args: unknown[]) => {
    console.warn('[WARN]', ...args);
  },
  error: (...args: unknown[]) => {
    console.error('[ERROR]', ...args);
  },
};
