/**
 * 保存策略配置对话框组件
 */

'use client';

import React, { useState, useEffect } from 'react';
import {
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  Button,
  Input,
  Textarea,
  Chip,
} from '@heroui/react';
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
    <Modal isOpen={isOpen} onClose={onClose} size="2xl">
      <ModalContent>
        {(onClose) => (
          <>
            <ModalHeader className="flex flex-col gap-1">
              <div className="flex items-center space-x-2">
                <Save className="w-5 h-5" />
                <span>保存策略配置</span>
              </div>
            </ModalHeader>
            <ModalBody>
              <div className="space-y-4">
                <div>
                  <p className="text-sm text-default-500 mb-2">策略名称</p>
                  <Chip variant="flat" color="secondary">{strategyName}</Chip>
                </div>

                <Input
                  label="配置名称"
                  placeholder="请输入配置名称"
                  value={configName}
                  onValueChange={setConfigName}
                  isRequired
                  isInvalid={!!error && !configName.trim()}
                  errorMessage={error && !configName.trim() ? error : undefined}
                />

                <Textarea
                  label="配置描述"
                  placeholder="请输入配置描述（可选）"
                  value={description}
                  onValueChange={setDescription}
                  minRows={2}
                />

                <div>
                  <p className="text-sm text-default-500 mb-2">参数预览</p>
                  <div className="bg-default-100 rounded-lg p-3">
                    <pre className="text-xs text-default-600 whitespace-pre-wrap font-mono">
                      {formatParameters()}
                    </pre>
                  </div>
                </div>

                {error && (
                  <div className="bg-danger-50 border border-danger-200 rounded-lg p-3">
                    <p className="text-sm text-danger">{error}</p>
                  </div>
                )}
              </div>
            </ModalBody>
            <ModalFooter>
              <Button variant="light" onPress={onClose} isDisabled={saving || loading}>
                取消
              </Button>
              <Button
                color="primary"
                onPress={handleSave}
                isLoading={saving || loading}
                startContent={!saving && !loading ? <Save className="w-4 h-4" /> : undefined}
              >
                保存
              </Button>
            </ModalFooter>
          </>
        )}
      </ModalContent>
    </Modal>
  );
}

