// frontend/src/components/ProgressDisplay.tsx

import React from 'react';
import { Box, Typography, Paper, LinearProgress, Stack } from '@mui/material';

interface ProgressDisplayProps {
  isInferencing: boolean;
  currentVideo: string | null;
  currentProgress: number; // 단일 비디오 진행률
  totalVideos: number;
  processedVideos: number;
}

const ProgressDisplay: React.FC<ProgressDisplayProps> = ({
  isInferencing,
  currentVideo,
  currentProgress,
  totalVideos,
  processedVideos,
}) => {
  // 항상 표시, 값이 없으면 0%로
  const overallProgress = totalVideos > 0 ? (processedVideos / totalVideos) * 100 : 0;

  return (
    <Box>
      <Stack spacing={1.5}>
          {/* 단일 비디오 클립 추론 진행률 */}
          <Box>
              <Typography variant="body2" component="div" color="text.secondary" noWrap title={currentVideo || ''}>
                  단일 클립 진행률: {currentVideo || '-'} {typeof currentProgress === 'number' && !isNaN(currentProgress) ? currentProgress.toFixed(0) : '0'}%
              </Typography>
              <LinearProgress variant="determinate" value={typeof currentProgress === 'number' && !isNaN(currentProgress) ? currentProgress : 0} />
          </Box>
          {/* 전체 클립 갯수 대비 진행률 */}
          <Box>
              <Typography variant="body2" component="div" color="text.secondary">
                  전체 비디오 진행률: {processedVideos || 0} / {totalVideos || 0}
              </Typography>
              <LinearProgress variant="determinate" value={overallProgress} />
          </Box>
      </Stack>
    </Box>
  );
};

export default ProgressDisplay;