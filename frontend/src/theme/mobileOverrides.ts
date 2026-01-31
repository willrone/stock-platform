import { Theme } from '@mui/material/styles';

/**
 * 移动端主题覆盖配置
 * 优化触摸交互和视觉体验
 */
export const getMobileOverrides = (theme: Theme) => ({
  components: {
    MuiTypography: {
      styleOverrides: {
        h4: {
          [theme.breakpoints.down('sm')]: {
            fontSize: '1.5rem', // 从 2.125rem 缩小
          },
        },
        h5: {
          [theme.breakpoints.down('sm')]: {
            fontSize: '1.25rem',
          },
        },
        h6: {
          [theme.breakpoints.down('sm')]: {
            fontSize: '1.125rem',
          },
        },
        body1: {
          [theme.breakpoints.down('sm')]: {
            fontSize: '0.95rem', // 稍微加大，移动端更易读
          },
        },
        body2: {
          [theme.breakpoints.down('sm')]: {
            fontSize: '0.875rem',
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          // 触摸设备最小点击区域 44x44px
          '@media (pointer: coarse)': {
            minHeight: 44,
            padding: '10px 16px',
          },
          [theme.breakpoints.down('sm')]: {
            fontSize: '0.95rem',
          },
        },
        sizeSmall: {
          '@media (pointer: coarse)': {
            minHeight: 40,
            padding: '8px 12px',
          },
        },
        sizeLarge: {
          '@media (pointer: coarse)': {
            minHeight: 48,
            padding: '12px 20px',
          },
        },
      },
    },
    MuiIconButton: {
      styleOverrides: {
        root: {
          '@media (pointer: coarse)': {
            minWidth: 44,
            minHeight: 44,
          },
        },
        sizeSmall: {
          '@media (pointer: coarse)': {
            minWidth: 40,
            minHeight: 40,
          },
        },
        sizeLarge: {
          '@media (pointer: coarse)': {
            minWidth: 48,
            minHeight: 48,
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          [theme.breakpoints.down('sm')]: {
            borderRadius: 12, // 移动端圆角更明显
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          [theme.breakpoints.down('sm')]: {
            fontSize: '0.8rem',
          },
        },
        sizeSmall: {
          [theme.breakpoints.down('sm')]: {
            fontSize: '0.75rem',
          },
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '@media (pointer: coarse)': {
            '& .MuiInputBase-input': {
              fontSize: '16px', // iOS 上避免自动缩放
              minHeight: 44,
            },
          },
        },
      },
    },
    MuiTableCell: {
      styleOverrides: {
        root: {
          [theme.breakpoints.down('md')]: {
            padding: '12px 8px', // 移动端减少 padding
            fontSize: '0.875rem',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          [theme.breakpoints.down('sm')]: {
            borderRadius: 12,
          },
        },
      },
    },
    MuiDialog: {
      styleOverrides: {
        paper: {
          [theme.breakpoints.down('sm')]: {
            margin: 16, // 移动端留更多边距
            maxHeight: 'calc(100% - 32px)',
          },
        },
      },
    },
    MuiDialogTitle: {
      styleOverrides: {
        root: {
          [theme.breakpoints.down('sm')]: {
            fontSize: '1.25rem',
            padding: '16px 20px',
          },
        },
      },
    },
    MuiDialogContent: {
      styleOverrides: {
        root: {
          [theme.breakpoints.down('sm')]: {
            padding: '16px 20px',
          },
        },
      },
    },
    MuiDialogActions: {
      styleOverrides: {
        root: {
          [theme.breakpoints.down('sm')]: {
            padding: '12px 20px 16px',
            '& > :not(:first-of-type)': {
              marginLeft: 12,
            },
          },
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          '& .MuiToolbar-root': {
            [theme.breakpoints.down('sm')]: {
              minHeight: 56, // 移动端降低高度
              paddingLeft: 12,
              paddingRight: 12,
            },
          },
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          [theme.breakpoints.down('sm')]: {
            width: '85%', // 移动端侧边栏占 85% 宽度
            maxWidth: 320,
          },
        },
      },
    },
  },
});
