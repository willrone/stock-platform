/**
 * 增强的数据文件表格组件
 *
 * 显示本地数据文件的详细信息，包括：
 * - 文件完整性状态
 * - 压缩比信息
 * - 高级筛选功能
 * - 批量操作
 * - 文件预览
 */

'use client';

import React, { useState, useMemo } from 'react';
import {
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  Chip,
  Button,
  Tooltip,
  TextField,
  Select,
  MenuItem,
  Dialog,
  DialogContent,
  DialogTitle,
  DialogActions,
  LinearProgress,
  Box,
  Typography,
  IconButton,
  Menu,
  TableContainer,
  Paper,
  InputAdornment,
  FormControl,
  InputLabel,
  CircularProgress,
} from '@mui/material';
import {
  FileText,
  Download,
  Trash2,
  Search,
  Filter,
  Eye,
  AlertTriangle,
  CheckCircle,
  Clock,
  HardDrive,
  Zap,
  MoreVertical,
} from 'lucide-react';

interface DataFile {
  file_path: string;
  stock_code: string;
  date_range: {
    start: string;
    end: string;
  };
  record_count: number;
  file_size: number;
  last_modified: string;
  integrity_status: string;
  compression_ratio: number;
  created_at?: string;
}

interface EnhancedDataFileTableProps {
  files: DataFile[];
  loading?: boolean;
  selectedFiles: Set<string>;
  onSelectionChange: (keys: Set<string>) => void;
  onDeleteFiles: (filePaths: string[]) => void;
  onRefresh: () => void;
}

// 文件操作菜单组件
function FileMenu({ file: _file, onDelete }: { file: DataFile; onDelete: () => void }) {
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const open = Boolean(anchorEl);

  const handleClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  return (
    <>
      <IconButton size="small" onClick={handleClick}>
        <MoreVertical size={16} />
      </IconButton>
      <Menu anchorEl={anchorEl} open={open} onClose={handleClose}>
        <MenuItem onClick={handleClose}>
          <Download size={16} style={{ marginRight: 8 }} />
          下载文件
        </MenuItem>
        <MenuItem
          onClick={() => {
            onDelete();
            handleClose();
          }}
          sx={{ color: 'error.main' }}
        >
          <Trash2 size={16} style={{ marginRight: 8 }} />
          删除文件
        </MenuItem>
      </Menu>
    </>
  );
}

export function EnhancedDataFileTable({
  files,
  loading = false,
  selectedFiles,
  onSelectionChange: _onSelectionChange,
  onDeleteFiles,
  onRefresh: _onRefresh,
}: EnhancedDataFileTableProps) {
  const [searchText, setSearchText] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [sizeFilter, setSizeFilter] = useState<string>('');
  const [dateFilter, setDateFilter] = useState<string>('');
  const [selectedFile, setSelectedFile] = useState<DataFile | null>(null);

  const [isPreviewOpen, setIsPreviewOpen] = useState(false);
  const [isDeleteOpen, setIsDeleteOpen] = useState(false);
  const onPreviewOpen = () => setIsPreviewOpen(true);
  const onPreviewClose = () => setIsPreviewOpen(false);
  const onDeleteOpen = () => setIsDeleteOpen(true);
  const onDeleteClose = () => setIsDeleteOpen(false);

  // 格式化文件大小
  const formatFileSize = (bytes: number) => {
    if (bytes === 0) {
      return '0 B';
    }
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // 获取文件状态
  const getFileStatus = (file: DataFile) => {
    switch (file.integrity_status) {
      case 'valid':
        return { color: 'success' as const, text: '完整', icon: CheckCircle };
      case 'corrupted':
        return { color: 'error' as const, text: '损坏', icon: AlertTriangle };
      case 'partial':
        return { color: 'warning' as const, text: '部分', icon: Clock };
      default:
        return { color: 'default' as const, text: '未知', icon: FileText };
    }
  };

  // 获取压缩比颜色（暂时未使用）
  // const getCompressionColor = (ratio: number) => {
  //   if (ratio > 0.8) {
  //     return 'text-success';
  //   }
  //   if (ratio > 0.6) {
  //     return 'text-warning';
  //   }
  //   return 'text-danger';
  // };

  // 获取文件新旧程度
  const getFileAge = (file: DataFile) => {
    const now = new Date();
    const lastModified = new Date(file.last_modified);
    const hoursDiff = (now.getTime() - lastModified.getTime()) / (1000 * 60 * 60);

    if (hoursDiff < 24) {
      return { color: 'success' as const, text: '最新' };
    } else if (hoursDiff < 72) {
      return { color: 'warning' as const, text: '较新' };
    } else {
      return { color: 'error' as const, text: '过期' };
    }
  };

  // 过滤文件
  const filteredFiles = useMemo(() => {
    return files.filter(file => {
      const matchesSearch = file.stock_code.toLowerCase().includes(searchText.toLowerCase());
      const matchesStatus = !statusFilter || file.integrity_status === statusFilter;

      let matchesSize = true;
      if (sizeFilter) {
        const sizeInMB = file.file_size / (1024 * 1024);
        switch (sizeFilter) {
          case 'small':
            matchesSize = sizeInMB < 10;
            break;
          case 'medium':
            matchesSize = sizeInMB >= 10 && sizeInMB < 100;
            break;
          case 'large':
            matchesSize = sizeInMB >= 100;
            break;
        }
      }

      let matchesDate = true;
      if (dateFilter) {
        const age = getFileAge(file);
        matchesDate = age.text === dateFilter;
      }

      return matchesSearch && matchesStatus && matchesSize && matchesDate;
    });
  }, [files, searchText, statusFilter, sizeFilter, dateFilter]);

  const handlePreviewFile = (file: DataFile) => {
    setSelectedFile(file);
    onPreviewOpen();
  };

  const handleDeleteSelected = () => {
    const filesToDelete = Array.from(selectedFiles);
    onDeleteFiles(filesToDelete);
    onDeleteClose();
  };

  const clearFilters = () => {
    setSearchText('');
    setStatusFilter('');
    setSizeFilter('');
    setDateFilter('');
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {/* 筛选控件 */}
      <Box
        sx={{
          display: 'flex',
          flexDirection: { xs: 'column', md: 'row' },
          gap: 2,
          p: 2,
          bgcolor: 'grey.50',
          borderRadius: 1,
        }}
      >
        <TextField
          placeholder="搜索股票代码"
          value={searchText}
          onChange={e => setSearchText(e.target.value)}
          size="small"
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <Search size={16} />
              </InputAdornment>
            ),
          }}
          sx={{ width: { md: 256 } }}
        />

        <FormControl size="small" sx={{ minWidth: { md: 160 } }}>
          <InputLabel>完整性状态</InputLabel>
          <Select
            value={statusFilter}
            label="完整性状态"
            onChange={e => setStatusFilter(e.target.value)}
          >
            <MenuItem value="">全部</MenuItem>
            <MenuItem value="valid">完整</MenuItem>
            <MenuItem value="corrupted">损坏</MenuItem>
            <MenuItem value="partial">部分</MenuItem>
          </Select>
        </FormControl>

        <FormControl size="small" sx={{ minWidth: { md: 160 } }}>
          <InputLabel>文件大小</InputLabel>
          <Select value={sizeFilter} label="文件大小" onChange={e => setSizeFilter(e.target.value)}>
            <MenuItem value="">全部</MenuItem>
            <MenuItem value="small">小于 10MB</MenuItem>
            <MenuItem value="medium">10MB - 100MB</MenuItem>
            <MenuItem value="large">大于 100MB</MenuItem>
          </Select>
        </FormControl>

        <FormControl size="small" sx={{ minWidth: { md: 160 } }}>
          <InputLabel>文件新旧</InputLabel>
          <Select value={dateFilter} label="文件新旧" onChange={e => setDateFilter(e.target.value)}>
            <MenuItem value="">全部</MenuItem>
            <MenuItem value="最新">最新</MenuItem>
            <MenuItem value="较新">较新</MenuItem>
            <MenuItem value="过期">过期</MenuItem>
          </Select>
        </FormControl>

        <Button variant="outlined" startIcon={<Filter size={16} />} onClick={clearFilters}>
          清除筛选
        </Button>
      </Box>

      {/* 操作栏 */}
      {selectedFiles.size > 0 && (
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            p: 1.5,
            bgcolor: 'primary.light',
            borderRadius: 1,
          }}
        >
          <Typography variant="body2" sx={{ fontWeight: 500, color: 'primary.main' }}>
            已选择 {selectedFiles.size} 个文件
          </Typography>
          <Button
            variant="outlined"
            color="error"
            size="small"
            startIcon={<Trash2 size={16} />}
            onClick={onDeleteOpen}
          >
            删除选中
          </Button>
        </Box>
      )}

      {/* 文件表格 */}
      <TableContainer component={Paper}>
        <Table aria-label="增强的数据文件列表" sx={{ minHeight: 400 }}>
          <TableHead>
            <TableRow>
              <TableCell>股票代码</TableCell>
              <TableCell>文件大小</TableCell>
              <TableCell>记录数</TableCell>
              <TableCell>压缩比</TableCell>
              <TableCell>数据范围</TableCell>
              <TableCell>完整性</TableCell>
              <TableCell>最后修改</TableCell>
              <TableCell>操作</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {loading ? (
              <TableRow>
                <TableCell colSpan={8} align="center">
                  <Box sx={{ py: 4 }}>
                    <CircularProgress size={24} />
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                      加载文件列表中...
                    </Typography>
                  </Box>
                </TableCell>
              </TableRow>
            ) : filteredFiles.length === 0 ? (
              <TableRow>
                <TableCell colSpan={8} align="center">
                  <Typography variant="body2" color="text.secondary" sx={{ py: 4 }}>
                    暂无数据文件
                  </Typography>
                </TableCell>
              </TableRow>
            ) : (
              filteredFiles.map(file => {
                const status = getFileStatus(file);
                const age = getFileAge(file);
                const StatusIcon = status.icon;

                return (
                  <TableRow key={file.file_path}>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <FileText size={16} color="#999" />
                        <Box>
                          <Typography variant="body2" sx={{ fontWeight: 500 }}>
                            {file.stock_code}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {file.file_path.split('/').pop()}
                          </Typography>
                        </Box>
                      </Box>
                    </TableCell>

                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <HardDrive size={12} color="#999" />
                        <Typography variant="body2" sx={{ fontWeight: 500 }}>
                          {formatFileSize(file.file_size)}
                        </Typography>
                      </Box>
                    </TableCell>

                    <TableCell>
                      <Typography variant="body2" sx={{ fontWeight: 500 }}>
                        {file.record_count.toLocaleString()}
                      </Typography>
                    </TableCell>

                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Zap size={12} color="#999" />
                        <Typography
                          variant="body2"
                          sx={{
                            fontWeight: 500,
                            color:
                              file.compression_ratio > 0.8
                                ? 'success.main'
                                : file.compression_ratio > 0.6
                                  ? 'warning.main'
                                  : 'error.main',
                          }}
                        >
                          {(file.compression_ratio * 100).toFixed(1)}%
                        </Typography>
                      </Box>
                    </TableCell>

                    <TableCell>
                      <Box>
                        <Typography variant="body2" sx={{ fontWeight: 500 }}>
                          {file.date_range.start}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          至 {file.date_range.end}
                        </Typography>
                      </Box>
                    </TableCell>

                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Chip
                          label={status.text}
                          color={status.color}
                          size="small"
                          icon={<StatusIcon size={12} />}
                        />
                      </Box>
                    </TableCell>

                    <TableCell>
                      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                        <Typography variant="body2">
                          {new Date(file.last_modified).toLocaleDateString()}
                        </Typography>
                        <Chip label={age.text} color={age.color} size="small" />
                      </Box>
                    </TableCell>

                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <Tooltip title="预览文件信息">
                          <IconButton size="small" onClick={() => handlePreviewFile(file)}>
                            <Eye size={16} />
                          </IconButton>
                        </Tooltip>

                        <FileMenu file={file} onDelete={() => onDeleteFiles([file.file_path])} />
                      </Box>
                    </TableCell>
                  </TableRow>
                );
              })
            )}
          </TableBody>
        </Table>
      </TableContainer>

      {/* 文件预览模态框 */}
      <Dialog open={isPreviewOpen} onClose={onPreviewClose} maxWidth="md" fullWidth>
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <FileText size={20} />
            <Typography variant="h6" component="span">
              文件详情 - {selectedFile?.stock_code}
            </Typography>
          </Box>
        </DialogTitle>
        <DialogContent>
          {selectedFile && (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {/* 基本信息 */}
              <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 2 }}>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    文件路径
                  </Typography>
                  <Typography
                    variant="body2"
                    sx={{ fontFamily: 'monospace', wordBreak: 'break-all', mt: 0.5 }}
                  >
                    {selectedFile.file_path}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    股票代码
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500, mt: 0.5 }}>
                    {selectedFile.stock_code}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    文件大小
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500, mt: 0.5 }}>
                    {formatFileSize(selectedFile.file_size)}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    记录数量
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500, mt: 0.5 }}>
                    {selectedFile.record_count.toLocaleString()}
                  </Typography>
                </Box>
              </Box>

              {/* 数据质量指标 */}
              <Box sx={{ p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 500, mb: 1.5 }}>
                  数据质量指标
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                  <Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                      <Typography variant="body2">压缩效率</Typography>
                      <Typography variant="body2" sx={{ fontWeight: 500 }}>
                        {(selectedFile.compression_ratio * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={selectedFile.compression_ratio * 100}
                      color={
                        selectedFile.compression_ratio > 0.8
                          ? 'success'
                          : selectedFile.compression_ratio > 0.6
                            ? 'warning'
                            : 'error'
                      }
                      sx={{ height: 8, borderRadius: 4 }}
                    />
                  </Box>

                  <Box
                    sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
                  >
                    <Typography variant="body2">完整性状态</Typography>
                    <Chip
                      label={getFileStatus(selectedFile).text}
                      color={getFileStatus(selectedFile).color}
                      size="small"
                    />
                  </Box>
                </Box>
              </Box>

              {/* 时间信息 */}
              <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 2 }}>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    数据开始日期
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500, mt: 0.5 }}>
                    {selectedFile.date_range.start}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    数据结束日期
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500, mt: 0.5 }}>
                    {selectedFile.date_range.end}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    最后修改时间
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500, mt: 0.5 }}>
                    {new Date(selectedFile.last_modified).toLocaleString()}
                  </Typography>
                </Box>
                {selectedFile.created_at && (
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      创建时间
                    </Typography>
                    <Typography variant="body2" sx={{ fontWeight: 500, mt: 0.5 }}>
                      {new Date(selectedFile.created_at).toLocaleString()}
                    </Typography>
                  </Box>
                )}
              </Box>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button variant="outlined" onClick={onPreviewClose}>
            关闭
          </Button>
          <Button variant="contained" color="primary" startIcon={<Download size={16} />}>
            下载文件
          </Button>
        </DialogActions>
      </Dialog>

      {/* 删除确认模态框 */}
      <Dialog open={isDeleteOpen} onClose={onDeleteClose}>
        <DialogTitle>确认删除</DialogTitle>
        <DialogContent>
          <Typography variant="body2">
            确定要删除选中的 {selectedFiles.size} 个数据文件吗？此操作不可撤销。
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button variant="outlined" onClick={onDeleteClose}>
            取消
          </Button>
          <Button variant="contained" color="error" onClick={handleDeleteSelected}>
            删除
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
