'use client';
import { Box, Skeleton as MuiSkeleton } from '@mui/material';

interface Props {
  rows?: number;
  height?: number;
}

export default function PageSkeleton({ rows = 6, height = 40 }: Props) {
  return (
    <Box sx={{ p: 2 }}>
      {Array.from({ length: rows }).map((_, i) => (
        <MuiSkeleton
          key={i}
          variant="rectangular"
          height={height}
          sx={{ mb: 1.5, borderRadius: 1 }}
        />
      ))}
    </Box>
  );
}
