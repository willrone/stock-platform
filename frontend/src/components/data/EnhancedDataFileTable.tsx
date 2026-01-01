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
  TableHeader,
  TableColumn,
  TableBody,
  TableRow,
  TableCell,
  Chip,
  Button,
  Tooltip,
  Input,
  Select,
  SelectItem,
  DatePicker,
  Dropdown,
  DropdownTrigger,
  DropdownMenu,
  DropdownItem,
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  useDisclosure,
  Progress,
} from '@heroui/react';
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

export function EnhancedDataFileTable({
  files,
  loading = false,
  selectedFiles,
  onSelectionChange,
  onDeleteFiles,
  onRefresh,
}: EnhancedDataFileTableProps) {
  const [searchText, setSearchText] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [sizeFilter, setSizeFilter] = useState<string>('');
  const [dateFilter, setDateFilter] = useState<string>('');
  const [selectedFile, setSelectedFile] = useState<DataFile | null>(null);
  
  const { isOpen: isPreviewOpen, onOpen: onPreviewOpen, onClose: onPreviewClose } = useDisclosure();
  const { isOpen: isDeleteOpen, onOpen: onDeleteOpen, onClose: onDeleteClose } = useDisclosure();

  // 格式化文件大小
  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
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
        return { color: 'danger' as const, text: '损坏', icon: AlertTriangle };
      case 'partial':
        return { color: 'warning' as const, text: '部分', icon: Clock };
      default:
        return { color: 'default' as const, text: '未知', icon: FileText };
    }
  };

  // 获取压缩比颜色
  const getCompressionColor = (ratio: number) => {
    if (ratio > 0.8) return 'text-success';
    if (ratio > 0.6) return 'text-warning';
    return 'text-danger';
  };

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
      return { color: 'danger' as const, text: '过期' };
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
    <div className="space-y-4">
      {/* 筛选控件 */}
      <div className="flex flex-col md:flex-row gap-4 p-4 bg-default-50 rounded-lg">
        <Input
          placeholder="搜索股票代码"
          startContent={<Search className="w-4 h-4" />}
          value={searchText}
          onValueChange={setSearchText}
          className="md:w-64"
        />
        
        <Select
          placeholder="完整性状态"
          selectedKeys={statusFilter ? [statusFilter] : []}
          onSelectionChange={(keys) => setStatusFilter(Array.from(keys)[0] as string || '')}
          className="md:w-40"
        >
          <SelectItem key="valid">完整</SelectItem>
          <SelectItem key="corrupted">损坏</SelectItem>
          <SelectItem key="partial">部分</SelectItem>
        </Select>
        
        <Select
          placeholder="文件大小"
          selectedKeys={sizeFilter ? [sizeFilter] : []}
          onSelectionChange={(keys) => setSizeFilter(Array.from(keys)[0] as string || '')}
          className="md:w-40"
        >
          <SelectItem key="small">小于 10MB</SelectItem>
          <SelectItem key="medium">10MB - 100MB</SelectItem>
          <SelectItem key="large">大于 100MB</SelectItem>
        </Select>
        
        <Select
          placeholder="文件新旧"
          selectedKeys={dateFilter ? [dateFilter] : []}
          onSelectionChange={(keys) => setDateFilter(Array.from(keys)[0] as string || '')}
          className="md:w-40"
        >
          <SelectItem key="最新">最新</SelectItem>
          <SelectItem key="较新">较新</SelectItem>
          <SelectItem key="过期">过期</SelectItem>
        </Select>
        
        <Button
          variant="light"
          startContent={<Filter className="w-4 h-4" />}
          onPress={clearFilters}
        >
          清除筛选
        </Button>
      </div>

      {/* 操作栏 */}
      {selectedFiles.size > 0 && (
        <div className="flex items-center justify-between p-3 bg-primary-50 rounded-lg">
          <span className="text-sm font-medium text-primary">
            已选择 {selectedFiles.size} 个文件
          </span>
          <Button
            color="danger"
            variant="light"
            size="sm"
            startContent={<Trash2 className="w-4 h-4" />}
            onPress={onDeleteOpen}
          >
            删除选中
          </Button>
        </div>
      )}

      {/* 文件表格 */}
      <Table
        aria-label="增强的数据文件列表"
        selectionMode="multiple"
        selectedKeys={selectedFiles}
        onSelectionChange={(keys) => onSelectionChange(new Set(Array.from(keys).map(String)))}
        classNames={{
          wrapper: "min-h-[400px]",
        }}
      >
        <TableHeader>
          <TableColumn>股票代码</TableColumn>
          <TableColumn>文件大小</TableColumn>
          <TableColumn>记录数</TableColumn>
          <TableColumn>压缩比</TableColumn>
          <TableColumn>数据范围</TableColumn>
          <TableColumn>完整性</TableColumn>
          <TableColumn>最后修改</TableColumn>
          <TableColumn>操作</TableColumn>
        </TableHeader>
        <TableBody
          isLoading={loading}
          loadingContent="加载文件列表中..."
          emptyContent="暂无数据文件"
        >
          {filteredFiles.map((file) => {
            const status = getFileStatus(file);
            const age = getFileAge(file);
            const StatusIcon = status.icon;
            
            return (
              <TableRow key={file.file_path}>
                <TableCell>
                  <div className="flex items-center space-x-2">
                    <FileText className="w-4 h-4 text-default-400" />
                    <div>
                      <p className="font-medium">{file.stock_code}</p>
                      <p className="text-xs text-default-500">
                        {file.file_path.split('/').pop()}
                      </p>
                    </div>
                  </div>
                </TableCell>
                
                <TableCell>
                  <div className="flex items-center space-x-2">
                    <HardDrive className="w-3 h-3 text-default-400" />
                    <span className="text-sm font-medium">
                      {formatFileSize(file.file_size)}
                    </span>
                  </div>
                </TableCell>
                
                <TableCell>
                  <span className="text-sm font-medium">
                    {file.record_count.toLocaleString()}
                  </span>
                </TableCell>
                
                <TableCell>
                  <div className="flex items-center space-x-2">
                    <Zap className="w-3 h-3 text-default-400" />
                    <span className={`text-sm font-medium ${getCompressionColor(file.compression_ratio)}`}>
                      {(file.compression_ratio * 100).toFixed(1)}%
                    </span>
                  </div>
                </TableCell>
                
                <TableCell>
                  <div className="text-sm">
                    <div className="font-medium">{file.date_range.start}</div>
                    <div className="text-default-500">至 {file.date_range.end}</div>
                  </div>
                </TableCell>
                
                <TableCell>
                  <div className="flex items-center space-x-2">
                    <Chip
                      color={status.color}
                      variant="flat"
                      size="sm"
                      startContent={<StatusIcon className="w-3 h-3" />}
                    >
                      {status.text}
                    </Chip>
                  </div>
                </TableCell>
                
                <TableCell>
                  <div className="space-y-1">
                    <div className="text-sm">
                      {new Date(file.last_modified).toLocaleDateString()}
                    </div>
                    <Chip
                      color={age.color}
                      variant="flat"
                      size="sm"
                    >
                      {age.text}
                    </Chip>
                  </div>
                </TableCell>
                
                <TableCell>
                  <div className="flex items-center space-x-1">
                    <Tooltip content="预览文件信息">
                      <Button
                        isIconOnly
                        variant="light"
                        size="sm"
                        onPress={() => handlePreviewFile(file)}
                      >
                        <Eye className="w-4 h-4" />
                      </Button>
                    </Tooltip>
                    
                    <Dropdown>
                      <DropdownTrigger>
                        <Button
                          isIconOnly
                          variant="light"
                          size="sm"
                        >
                          <MoreVertical className="w-4 h-4" />
                        </Button>
                      </DropdownTrigger>
                      <DropdownMenu>
                        <DropdownItem
                          key="download"
                          startContent={<Download className="w-4 h-4" />}
                        >
                          下载文件
                        </DropdownItem>
                        <DropdownItem
                          key="delete"
                          className="text-danger"
                          color="danger"
                          startContent={<Trash2 className="w-4 h-4" />}
                          onPress={() => onDeleteFiles([file.file_path])}
                        >
                          删除文件
                        </DropdownItem>
                      </DropdownMenu>
                    </Dropdown>
                  </div>
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>

      {/* 文件预览模态框 */}
      <Modal isOpen={isPreviewOpen} onClose={onPreviewClose} size="2xl">
        <ModalContent>
          {(onClose) => (
            <>
              <ModalHeader>
                <div className="flex items-center space-x-2">
                  <FileText className="w-5 h-5" />
                  <span>文件详情 - {selectedFile?.stock_code}</span>
                </div>
              </ModalHeader>
              <ModalBody>
                {selectedFile && (
                  <div className="space-y-4">
                    {/* 基本信息 */}
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-sm text-default-500">文件路径</p>
                        <p className="font-mono text-sm break-all">{selectedFile.file_path}</p>
                      </div>
                      <div>
                        <p className="text-sm text-default-500">股票代码</p>
                        <p className="font-medium">{selectedFile.stock_code}</p>
                      </div>
                      <div>
                        <p className="text-sm text-default-500">文件大小</p>
                        <p className="font-medium">{formatFileSize(selectedFile.file_size)}</p>
                      </div>
                      <div>
                        <p className="text-sm text-default-500">记录数量</p>
                        <p className="font-medium">{selectedFile.record_count.toLocaleString()}</p>
                      </div>
                    </div>

                    {/* 数据质量指标 */}
                    <div className="p-4 bg-default-50 rounded-lg">
                      <h4 className="font-medium mb-3">数据质量指标</h4>
                      <div className="space-y-3">
                        <div>
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-sm">压缩效率</span>
                            <span className="text-sm font-medium">
                              {(selectedFile.compression_ratio * 100).toFixed(1)}%
                            </span>
                          </div>
                          <Progress
                            value={selectedFile.compression_ratio * 100}
                            color={selectedFile.compression_ratio > 0.8 ? 'success' : 
                                   selectedFile.compression_ratio > 0.6 ? 'warning' : 'danger'}
                            className="w-full"
                          />
                        </div>
                        
                        <div className="flex items-center justify-between">
                          <span className="text-sm">完整性状态</span>
                          <Chip
                            color={getFileStatus(selectedFile).color}
                            variant="flat"
                            size="sm"
                          >
                            {getFileStatus(selectedFile).text}
                          </Chip>
                        </div>
                      </div>
                    </div>

                    {/* 时间信息 */}
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-sm text-default-500">数据开始日期</p>
                        <p className="font-medium">{selectedFile.date_range.start}</p>
                      </div>
                      <div>
                        <p className="text-sm text-default-500">数据结束日期</p>
                        <p className="font-medium">{selectedFile.date_range.end}</p>
                      </div>
                      <div>
                        <p className="text-sm text-default-500">最后修改时间</p>
                        <p className="font-medium">
                          {new Date(selectedFile.last_modified).toLocaleString()}
                        </p>
                      </div>
                      {selectedFile.created_at && (
                        <div>
                          <p className="text-sm text-default-500">创建时间</p>
                          <p className="font-medium">
                            {new Date(selectedFile.created_at).toLocaleString()}
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </ModalBody>
              <ModalFooter>
                <Button variant="light" onPress={onClose}>
                  关闭
                </Button>
                <Button
                  color="primary"
                  startContent={<Download className="w-4 h-4" />}
                >
                  下载文件
                </Button>
              </ModalFooter>
            </>
          )}
        </ModalContent>
      </Modal>

      {/* 删除确认模态框 */}
      <Modal isOpen={isDeleteOpen} onClose={onDeleteClose}>
        <ModalContent>
          {(onClose) => (
            <>
              <ModalHeader>确认删除</ModalHeader>
              <ModalBody>
                <p>确定要删除选中的 {selectedFiles.size} 个数据文件吗？此操作不可撤销。</p>
              </ModalBody>
              <ModalFooter>
                <Button variant="light" onPress={onClose}>
                  取消
                </Button>
                <Button color="danger" onPress={handleDeleteSelected}>
                  删除
                </Button>
              </ModalFooter>
            </>
          )}
        </ModalContent>
      </Modal>
    </div>
  );
}