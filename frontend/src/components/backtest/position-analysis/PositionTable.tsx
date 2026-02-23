/**
 * 持仓数据表格组件
 */

import React from 'react';
import {
  Card,
  CardContent,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  TableContainer,
  Paper,
  TableSortLabel,
  LinearProgress,
  Chip,
  Button,
  Box,
  Typography,
} from '@mui/material';
import { formatCurrency, formatPercent } from '@/utils/backtest/formatters';
import { PositionData, SortConfig } from '@/utils/backtest/positionDataUtils';

interface PositionTableProps {
  sortedPositions: PositionData[];
  sortConfig: SortConfig;
  onSort: (key: keyof PositionData) => void;
  onStockClick: (stock: PositionData) => void;
}

export const PositionTable: React.FC<PositionTableProps> = ({
  sortedPositions,
  sortConfig,
  onSort,
  onStockClick,
}) => {
  return (
    <Card>
      <CardContent sx={{ p: { xs: 1, md: 0 } }}>
        <Box sx={{ overflowX: 'auto' }}>
          <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 600 }}>
          <Table stickyHeader>
            <TableHead>
              <TableRow>
                <TableCell>
                  <TableSortLabel
                    active={sortConfig.key === 'stock_code'}
                    direction={sortConfig.key === 'stock_code' ? sortConfig.direction : 'asc'}
                    onClick={() => onSort('stock_code')}
                  >
                    股票代码
                  </TableSortLabel>
                </TableCell>
                <TableCell align="right">
                  <TableSortLabel
                    active={sortConfig.key === 'total_return'}
                    direction={sortConfig.key === 'total_return' ? sortConfig.direction : 'asc'}
                    onClick={() => onSort('total_return')}
                  >
                    总收益
                  </TableSortLabel>
                </TableCell>
                <TableCell align="right">
                  <TableSortLabel
                    active={sortConfig.key === 'trade_count'}
                    direction={sortConfig.key === 'trade_count' ? sortConfig.direction : 'asc'}
                    onClick={() => onSort('trade_count')}
                  >
                    交易次数
                  </TableSortLabel>
                </TableCell>
                <TableCell align="right">
                  <TableSortLabel
                    active={sortConfig.key === 'win_rate'}
                    direction={sortConfig.key === 'win_rate' ? sortConfig.direction : 'asc'}
                    onClick={() => onSort('win_rate')}
                  >
                    胜率
                  </TableSortLabel>
                </TableCell>
                <TableCell align="right">
                  <TableSortLabel
                    active={sortConfig.key === 'avg_holding_period'}
                    direction={
                      sortConfig.key === 'avg_holding_period' ? sortConfig.direction : 'asc'
                    }
                    onClick={() => onSort('avg_holding_period')}
                  >
                    平均持仓期
                  </TableSortLabel>
                </TableCell>
                <TableCell>表现</TableCell>
                <TableCell>操作</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {sortedPositions.map(position => (
                <TableRow key={position.stock_code} hover>
                  <TableCell>
                    <Box>
                      <Typography
                        variant="body2"
                        sx={{ fontFamily: 'monospace', fontWeight: 600 }}
                      >
                        {position.stock_code}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {position.stock_name}
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell align="right">
                    <Typography
                      variant="body2"
                      sx={{
                        fontFamily: 'monospace',
                        fontWeight: 500,
                        color:
                          position.total_return > 0
                            ? 'success.main'
                            : position.total_return < 0
                              ? 'error.main'
                              : 'text.secondary',
                      }}
                    >
                      {formatCurrency(position.total_return)}
                    </Typography>
                  </TableCell>
                  <TableCell align="right">
                    <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                      {position.trade_count}
                    </Typography>
                  </TableCell>
                  <TableCell align="right">
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <LinearProgress
                        variant="determinate"
                        value={position.win_rate * 100}
                        sx={{ width: 60, height: 8, borderRadius: 4 }}
                        color={position.win_rate >= 0.5 ? 'success' : 'error'}
                      />
                      <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
                        {formatPercent(position.win_rate)}
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell align="right">
                    <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                      {position.avg_holding_period} 天
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={`${position.winning_trades}胜${position.losing_trades}负`}
                      size="small"
                      color={position.winning_trades > position.losing_trades ? 'success' : 'error'}
                    />
                  </TableCell>
                  <TableCell>
                    <Button
                      size="small"
                      variant="outlined"
                      onClick={() => onStockClick(position)}
                    >
                      详情
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
        </Box>
      </CardContent>
    </Card>
  );
};
