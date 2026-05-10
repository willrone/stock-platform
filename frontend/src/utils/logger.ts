type LogArgs = unknown[];

const isDevelopment = process.env.NODE_ENV !== 'production';

export const logger = {
  debug: (...args: LogArgs) => {
    if (isDevelopment) {
      globalThis.console.log(...args);
    }
  },
  info: (...args: LogArgs) => {
    if (isDevelopment) {
      globalThis.console.info(...args);
    }
  },
  warn: (...args: LogArgs) => {
    globalThis.console.warn(...args);
  },
  error: (...args: LogArgs) => {
    globalThis.console.error(...args);
  },
};
