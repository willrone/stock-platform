'use client';
import { Box, Typography, Button } from '@mui/material';

interface EmptyStateProps {
  icon?: string;
  title: string;
  description?: string;
  actionLabel?: string;
  onAction?: () => void;
}

export default function EmptyState({ icon = '📭', title, description, actionLabel, onAction }: EmptyStateProps) {
  return (
    <Box sx={{ textAlign: 'center', py: 8, px: 2 }}>
      <Typography variant="h3" sx={{ mb: 2 }}>{icon}</Typography>
      <Typography variant="h6" color="text.secondary" sx={{ mb: 1 }}>{title}</Typography>
      {description && <Typography variant="body2" color="text.disabled" sx={{ mb: 3 }}>{description}</Typography>}
      {actionLabel && onAction && (
        <Button variant="contained" onClick={onAction}>{actionLabel}</Button>
      )}
    </Box>
  );
}
