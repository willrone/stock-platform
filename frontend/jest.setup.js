import '@testing-library/jest-dom';

// Mock ResizeObserver
global.ResizeObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

// Mock IntersectionObserver
global.IntersectionObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

// Mock window.matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(), // deprecated
    removeListener: jest.fn(), // deprecated
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

// Mock framer-motion completely
jest.mock('framer-motion', () => ({
  motion: new Proxy({}, {
    get: () => 'div'
  }),
  AnimatePresence: ({ children }) => children,
  useAnimation: () => ({
    start: jest.fn(),
    stop: jest.fn(),
    set: jest.fn(),
  }),
  useMotionValue: () => ({
    get: jest.fn(),
    set: jest.fn(),
  }),
  useTransform: () => jest.fn(),
  useSpring: () => jest.fn(),
}));

// Mock ECharts
jest.mock('echarts', () => ({
  init: jest.fn(() => ({
    setOption: jest.fn(),
    dispose: jest.fn(),
    resize: jest.fn(),
    on: jest.fn(),
    off: jest.fn(),
  })),
  graphic: {
    LinearGradient: jest.fn(),
  },
  registerTheme: jest.fn(),
}));

/*
 * 【P1 · design】HeroUI 已移除（全站 0 引用，只注入 128KB 死 CSS），
 * 对应的 jest mock 同步删除 —— 否则 jest.mock('@heroui/react') 会因模块不存在而失败。
 */