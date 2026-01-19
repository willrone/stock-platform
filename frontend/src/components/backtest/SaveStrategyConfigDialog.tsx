/**
 * 保存策略配置对话框组件
 */

'use client';

import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogActions,
  Button,
  TextField,
  Chip,
  Box,
  Typography,
  Alert,
} from '@mui/material';
import { Save } from 'lucide-react';

export interface SaveStrategyConfigDialogProps {
  isOpen: boolean;
  onClose: () => void;
  strategyName: string;
  parameters: Record<string, any>;
  onSave: (configName: string, description: string) => Promise<void>;
  loading?: boolean;
}

export function SaveStrategyConfigDialog({
  isOpen,
  onClose,
  strategyName,
  parameters,
  onSave,
  loading = false,
}: SaveStrategyConfigDialogProps) {
  const [configName, setConfigName] = useState('');
  const [description, setDescription] = useState('');
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);

  // 当对话框打开时，生成默认配置名称
  useEffect(() => {
    if (isOpen) {
      const defaultName = `${strategyName}_${new Date().toISOString().split('T')[0]}`;
      setConfigName(defaultName);
      setDescription('');
      setError('');
    }
  }, [isOpen, strategyName]);

  // 处理保存
  const handleSave = async () => {
    if (!configName.trim()) {
      setError('请输入配置名称');
      return;
    }

    setError('');
    setSaving(true);

    try {
      await onSave(configName.trim(), description.trim());
      onClose();
      setConfigName('');
      setDescription('');
    } catch (err: any) {
      setError(err.message || '保存配置失败');
    } finally {
      setSaving(false);
    }
  };

  // 格式化参数预览
  const formatParameters = () => {
    return Object.entries(parameters)
      .map(([key, value]) => {
        if (typeof value === 'object' && value !== null) {
          return `${key}: ${JSON.stringify(value)}`;
        }
        return `${key}: ${value}`;
      })
      .join('\n');
  };

  return (
    <Dialog open={isOpen} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Save size={20} />
          <span>保存策略配置</span>
        </Box>
      </DialogTitle>
      <DialogContent>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <Box>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              策略名称
            </Typography>
            <Chip label={strategyName} color="secondary" size="small" />
          </Box>

          <TextField
            label="配置名称"
            placeholder="请输入配置名称"
            value={configName}
            onChange={(e) => setConfigName(e.target.value)}
            required
            error={!!error && !configName.trim()}
            helperText={error && !configName.trim() ? error : undefined}
            fullWidth
          />

          <TextField
            label="配置描述"
            placeholder="请输入配置描述（可选）"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            multiline
            rows={2}
            fullWidth
          />

          <Box>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              参数预览
            </Typography>
            <Box sx={{ bgcolor: 'grey.100', borderRadius: 1, p: 2 }}>
              <Typography
                variant="caption"
                component="pre"
                sx={{
                  fontFamily: 'monospace',
                  whiteSpace: 'pre-wrap',
                  m: 0,
                  fontSize: '0.75rem',
                }}
              >
                {formatParameters()}
              </Typography>
            </Box>
          </Box>

          {error && (
            <Alert severity="error">{error}</Alert>
          )}
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} disabled={saving || loading}>
          取消
        </Button>
        <Button
          variant="contained"
          color="primary"
          onClick={handleSave}
          disabled={saving || loading}
          startIcon={!saving && !loading ? <Save size={16} /> : undefined}
        >
          {saving || loading ? '保存中...' : '保存'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
