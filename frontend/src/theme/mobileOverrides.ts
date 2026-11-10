import { Theme } from '@mui/material/styles';
import { mobileTypography, radii, spacing } from './tokens';

/**
 * 移动端主题覆盖配置
 * 优化触摸交互和视觉体验
 *
 * 数值全部来自 `src/theme/tokens.ts`（唯一真相源），这里只做断点编排。
 */
export const getMobileOverrides = (theme: Theme) => ({
  components: {
    MuiTypography: {
      styleOverrides: {
        h4: {
          [theme.breakpoints.down('sm')]: {
            fontSize: mobileTypography.h4.fontSize,
          },
        },
        h5: {
          [theme.breakpoints.down('sm')]: {
            fontSize: mobileTypography.h5.fontSize,
          },
        },
        h6: {
          [theme.breakpoints.down('sm')]: {
            fontSize: mobileTypography.h6.fontSize,
          },
        },
        body1: {
          [theme.breakpoints.down('sm')]: {
            fontSize: mobileTypography.body1.fontSize,
          },
        },
        body2: {
          [theme.breakpoints.down('sm')]: {
            fontSize: mobileTypography.body2.fontSize,
          },
        },
        caption: {
          [theme.breakpoints.down('sm')]: {
            fontSize: mobileTypography.caption.fontSize,
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
            padding: `${spacing.sm}px ${spacing.lg}px`,
          },
          [theme.breakpoints.down('sm')]: {
            fontSize: mobileTypography.body1.fontSize,
          },
        },
        sizeSmall: {
          '@media (pointer: coarse)': {
            minHeight: 40,
            padding: `${spacing.xs}px ${spacing.md}px`,
          },
        },
        sizeLarge: {
          '@media (pointer: coarse)': {
            minHeight: 48,
            padding: `${spacing.md}px ${spacing.xl}px`,
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
            borderRadius: radii.lg, // 移动端圆角更明显
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          [theme.breakpoints.down('sm')]: {
            fontSize: mobileTypography.caption.fontSize,
          },
        },
        sizeSmall: {
          [theme.breakpoints.down('sm')]: {
            fontSize: mobileTypography.overline.fontSize,
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
            padding: `${spacing.md}px ${spacing.sm}px`, // 移动端减少 padding
            fontSize: mobileTypography.body2.fontSize,
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          [theme.breakpoints.down('sm')]: {
            borderRadius: radii.lg,
          },
        },
      },
    },
    MuiDialog: {
      styleOverrides: {
        paper: {
          [theme.breakpoints.down('sm')]: {
            margin: spacing.lg, // 移动端留更多边距
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
            padding: `${spacing.lg}px ${spacing.xl}px`,
          },
        },
      },
    },
    MuiDialogContent: {
      styleOverrides: {
        root: {
          [theme.breakpoints.down('sm')]: {
            padding: `${spacing.lg}px ${spacing.xl}px`,
          },
        },
      },
    },
    MuiDialogActions: {
      styleOverrides: {
        root: {
          [theme.breakpoints.down('sm')]: {
            padding: `${spacing.md}px ${spacing.xl}px ${spacing.lg}px`,
            '& > :not(:first-of-type)': {
              marginLeft: spacing.md,
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
              paddingLeft: spacing.md,
              paddingRight: spacing.md,
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
