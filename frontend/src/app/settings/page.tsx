'use client';

import React, { useState } from 'react';
import {
  Card,
  CardContent,
  Button,
  TextField,
  Switch,
  FormControlLabel,
  Tabs,
  Tab,
  Box,
  Typography,
  Divider,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  IconButton,
  Snackbar,
  Alert,
} from '@mui/material';
import { Settings, Database, Bell, Server, Save, RefreshCw, Palette } from 'lucide-react';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`settings-tabpanel-${index}`}
      aria-labelledby={`settings-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

export default function SettingsPage() {
  const [tabValue, setTabValue] = useState(0);
  const [loading, setLoading] = useState(false);
  const [snackbar, setSnackbar] = useState<{ open: boolean; message: string; severity: 'success' | 'error' }>({
    open: false,
    message: '',
    severity: 'success',
  });

  // 通用设置
  const [generalSettings, setGeneralSettings] = useState({
    systemName: '股票预测平台',
    language: 'zh-CN',
    timezone: 'Asia/Shanghai',
    dateFormat: 'YYYY-MM-DD',
  });

  // 数据设置
  const [dataSettings, setDataSettings] = useState({
    dataServiceUrl: 'http://localhost:5002',
    autoSync: true,
    syncInterval: 60,
    cacheEnabled: true,
    cacheTTL: 3600,
  });

  // 通知设置
  const [notificationSettings, setNotificationSettings] = useState({
    emailEnabled: false,
    emailAddress: '',
    taskCompleteNotify: true,
    taskFailNotify: true,
    dailyReport: false,
  });

  // 性能设置
  const [performanceSettings, setPerformanceSettings] = useState({
    maxWorkers: 4,
    batchSize: 100,
    enableProfiling: false,
    logLevel: 'INFO',
  });

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const showMessage = (message: string, severity: 'success' | 'error') => {
    setSnackbar({ open: true, message, severity });
  };

  const handleCloseSnackbar = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  const handleSaveGeneral = async () => {
    setLoading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 500));
      showMessage('通用设置已保存', 'success');
    } catch (error) {
      showMessage('保存失败', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleSaveData = async () => {
    setLoading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 500));
      showMessage('数据设置已保存', 'success');
    } catch (error) {
      showMessage('保存失败', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleSaveNotification = async () => {
    setLoading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 500));
      showMessage('通知设置已保存', 'success');
    } catch (error) {
      showMessage('保存失败', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleSavePerformance = async () => {
    setLoading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 500));
      showMessage('性能设置已保存', 'success');
    } catch (error) {
      showMessage('保存失败', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleTestConnection = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${dataSettings.dataServiceUrl}/health`);
      if (response.ok) {
        showMessage('数据服务连接正常', 'success');
      } else {
        showMessage('数据服务连接失败', 'error');
      }
    } catch (error) {
      showMessage('无法连接到数据服务', 'error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ p: { xs: 1.5, sm: 2, md: 3 } }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
        <Settings size={32} />
        <Box>
          <Typography variant="h4" fontWeight="bold" sx={{ fontSize: { xs: '1.5rem', sm: '1.75rem', md: '2.125rem' } }}>系统设置</Typography>
          <Typography variant="body2" color="text.secondary">配置系统参数和偏好设置</Typography>
        </Box>
      </Box>

      <Card>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={handleTabChange} aria-label="settings tabs" variant="scrollable" scrollButtons="auto">
            <Tab icon={<Palette size={18} />} iconPosition="start" label="通用设置" />
            <Tab icon={<Database size={18} />} iconPosition="start" label="数据设置" />
            <Tab icon={<Bell size={18} />} iconPosition="start" label="通知设置" />
            <Tab icon={<Server size={18} />} iconPosition="start" label="性能设置" />
          </Tabs>
        </Box>

        {/* 通用设置 */}
        <TabPanel value={tabValue} index={0}>
          <CardContent>
            <Typography variant="h6" gutterBottom>通用设置</Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              配置系统的基本参数
            </Typography>

            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' }, gap: 3 }}>
              <Box>
                <TextField
                  fullWidth
                  label="系统名称"
                  value={generalSettings.systemName}
                  onChange={(e) => setGeneralSettings({ ...generalSettings, systemName: e.target.value })}
                />
              </Box>
              <Box>
                <FormControl fullWidth>
                  <InputLabel>语言</InputLabel>
                  <Select
                    value={generalSettings.language}
                    label="语言"
                    onChange={(e) => setGeneralSettings({ ...generalSettings, language: e.target.value })}
                  >
                    <MenuItem value="zh-CN">简体中文</MenuItem>
                    <MenuItem value="en-US">English</MenuItem>
                  </Select>
                </FormControl>
              </Box>
              <Box>
                <FormControl fullWidth>
                  <InputLabel>时区</InputLabel>
                  <Select
                    value={generalSettings.timezone}
                    label="时区"
                    onChange={(e) => setGeneralSettings({ ...generalSettings, timezone: e.target.value })}
                  >
                    <MenuItem value="Asia/Shanghai">Asia/Shanghai (UTC+8)</MenuItem>
                    <MenuItem value="Asia/Hong_Kong">Asia/Hong_Kong (UTC+8)</MenuItem>
                    <MenuItem value="UTC">UTC</MenuItem>
                  </Select>
                </FormControl>
              </Box>
              <Box>
                <FormControl fullWidth>
                  <InputLabel>日期格式</InputLabel>
                  <Select
                    value={generalSettings.dateFormat}
                    label="日期格式"
                    onChange={(e) => setGeneralSettings({ ...generalSettings, dateFormat: e.target.value })}
                  >
                    <MenuItem value="YYYY-MM-DD">YYYY-MM-DD</MenuItem>
                    <MenuItem value="DD/MM/YYYY">DD/MM/YYYY</MenuItem>
                    <MenuItem value="MM/DD/YYYY">MM/DD/YYYY</MenuItem>
                  </Select>
                </FormControl>
              </Box>
            </Box>

            <Divider sx={{ my: 3 }} />

            <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
              <Button
                variant="contained"
                startIcon={<Save size={18} />}
                onClick={handleSaveGeneral}
                disabled={loading}
              >
                保存设置
              </Button>
            </Box>
          </CardContent>
        </TabPanel>

        {/* 数据设置 */}
        <TabPanel value={tabValue} index={1}>
          <CardContent>
            <Typography variant="h6" gutterBottom>数据设置</Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              配置数据服务和缓存参数
            </Typography>

            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' }, gap: 3 }}>
              <Box>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <TextField
                    fullWidth
                    label="数据服务地址"
                    value={dataSettings.dataServiceUrl}
                    onChange={(e) => setDataSettings({ ...dataSettings, dataServiceUrl: e.target.value })}
                  />
                  <IconButton onClick={handleTestConnection} disabled={loading} color="primary">
                    <RefreshCw size={20} />
                  </IconButton>
                </Box>
              </Box>
              <Box>
                <TextField
                  fullWidth
                  type="number"
                  label="同步间隔（分钟）"
                  value={dataSettings.syncInterval}
                  onChange={(e) => setDataSettings({ ...dataSettings, syncInterval: parseInt(e.target.value) })}
                />
              </Box>
            </Box>

            <Box sx={{ mt: 3 }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={dataSettings.autoSync}
                    onChange={(e) => setDataSettings({ ...dataSettings, autoSync: e.target.checked })}
                  />
                }
                label={
                  <Box>
                    <Typography>自动同步</Typography>
                    <Typography variant="caption" color="text.secondary">定期从远端同步股票数据</Typography>
                  </Box>
                }
              />
            </Box>

            <Box sx={{ mt: 2 }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={dataSettings.cacheEnabled}
                    onChange={(e) => setDataSettings({ ...dataSettings, cacheEnabled: e.target.checked })}
                  />
                }
                label={
                  <Box>
                    <Typography>启用缓存</Typography>
                    <Typography variant="caption" color="text.secondary">缓存查询结果以提高性能</Typography>
                  </Box>
                }
              />
            </Box>

            {dataSettings.cacheEnabled && (
              <Box sx={{ mt: 2, ml: 4 }}>
                <TextField
                  type="number"
                  label="缓存过期时间（秒）"
                  value={dataSettings.cacheTTL}
                  onChange={(e) => setDataSettings({ ...dataSettings, cacheTTL: parseInt(e.target.value) })}
                  sx={{ width: 200 }}
                />
              </Box>
            )}

            <Divider sx={{ my: 3 }} />

            <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
              <Button
                variant="contained"
                startIcon={<Save size={18} />}
                onClick={handleSaveData}
                disabled={loading}
              >
                保存设置
              </Button>
            </Box>
          </CardContent>
        </TabPanel>

        {/* 通知设置 */}
        <TabPanel value={tabValue} index={2}>
          <CardContent>
            <Typography variant="h6" gutterBottom>通知设置</Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              配置系统通知和提醒
            </Typography>

            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={notificationSettings.emailEnabled}
                    onChange={(e) => setNotificationSettings({ ...notificationSettings, emailEnabled: e.target.checked })}
                  />
                }
                label={
                  <Box>
                    <Typography>邮件通知</Typography>
                    <Typography variant="caption" color="text.secondary">通过邮件接收系统通知</Typography>
                  </Box>
                }
              />

              {notificationSettings.emailEnabled && (
                <Box sx={{ ml: 4, mt: 1 }}>
                  <TextField
                    type="email"
                    label="邮箱地址"
                    placeholder="your@email.com"
                    value={notificationSettings.emailAddress}
                    onChange={(e) => setNotificationSettings({ ...notificationSettings, emailAddress: e.target.value })}
                    sx={{ width: 300 }}
                  />
                </Box>
              )}

              <Divider />

              <FormControlLabel
                control={
                  <Switch
                    checked={notificationSettings.taskCompleteNotify}
                    onChange={(e) => setNotificationSettings({ ...notificationSettings, taskCompleteNotify: e.target.checked })}
                  />
                }
                label={
                  <Box>
                    <Typography>任务完成通知</Typography>
                    <Typography variant="caption" color="text.secondary">任务完成时发送通知</Typography>
                  </Box>
                }
              />

              <FormControlLabel
                control={
                  <Switch
                    checked={notificationSettings.taskFailNotify}
                    onChange={(e) => setNotificationSettings({ ...notificationSettings, taskFailNotify: e.target.checked })}
                  />
                }
                label={
                  <Box>
                    <Typography>任务失败通知</Typography>
                    <Typography variant="caption" color="text.secondary">任务失败时发送通知</Typography>
                  </Box>
                }
              />

              <FormControlLabel
                control={
                  <Switch
                    checked={notificationSettings.dailyReport}
                    onChange={(e) => setNotificationSettings({ ...notificationSettings, dailyReport: e.target.checked })}
                  />
                }
                label={
                  <Box>
                    <Typography>每日报告</Typography>
                    <Typography variant="caption" color="text.secondary">每天发送系统运行报告</Typography>
                  </Box>
                }
              />
            </Box>

            <Divider sx={{ my: 3 }} />

            <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
              <Button
                variant="contained"
                startIcon={<Save size={18} />}
                onClick={handleSaveNotification}
                disabled={loading}
              >
                保存设置
              </Button>
            </Box>
          </CardContent>
        </TabPanel>

        {/* 性能设置 */}
        <TabPanel value={tabValue} index={3}>
          <CardContent>
            <Typography variant="h6" gutterBottom>性能设置</Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              配置系统性能参数
            </Typography>

            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' }, gap: 3 }}>
              <Box>
                <TextField
                  fullWidth
                  type="number"
                  label="最大工作进程数"
                  value={performanceSettings.maxWorkers}
                  onChange={(e) => setPerformanceSettings({ ...performanceSettings, maxWorkers: parseInt(e.target.value) })}
                  helperText="建议设置为 CPU 核心数"
                  inputProps={{ min: 1, max: 16 }}
                />
              </Box>
              <Box>
                <TextField
                  fullWidth
                  type="number"
                  label="批处理大小"
                  value={performanceSettings.batchSize}
                  onChange={(e) => setPerformanceSettings({ ...performanceSettings, batchSize: parseInt(e.target.value) })}
                  helperText="每批处理的股票数量"
                  inputProps={{ min: 10, max: 1000 }}
                />
              </Box>
              <Box>
                <FormControl fullWidth>
                  <InputLabel>日志级别</InputLabel>
                  <Select
                    value={performanceSettings.logLevel}
                    label="日志级别"
                    onChange={(e) => setPerformanceSettings({ ...performanceSettings, logLevel: e.target.value })}
                  >
                    <MenuItem value="DEBUG">DEBUG</MenuItem>
                    <MenuItem value="INFO">INFO</MenuItem>
                    <MenuItem value="WARNING">WARNING</MenuItem>
                    <MenuItem value="ERROR">ERROR</MenuItem>
                  </Select>
                </FormControl>
              </Box>
            </Box>

            <Box sx={{ mt: 3 }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={performanceSettings.enableProfiling}
                    onChange={(e) => setPerformanceSettings({ ...performanceSettings, enableProfiling: e.target.checked })}
                  />
                }
                label={
                  <Box>
                    <Typography>性能分析</Typography>
                    <Typography variant="caption" color="text.secondary">启用性能分析（会影响执行速度）</Typography>
                  </Box>
                }
              />
            </Box>

            <Divider sx={{ my: 3 }} />

            <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
              <Button
                variant="contained"
                startIcon={<Save size={18} />}
                onClick={handleSavePerformance}
                disabled={loading}
              >
                保存设置
              </Button>
            </Box>
          </CardContent>
        </TabPanel>
      </Card>

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={3000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert onClose={handleCloseSnackbar} severity={snackbar.severity} sx={{ width: '100%' }}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
}
