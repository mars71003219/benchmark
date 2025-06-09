import React from 'react';
import { Box, Typography, Paper } from '@mui/material';

const ModelLoadedDisplay: React.FC<{ modelId: string }> = ({ modelId }) => {
  return (
    <Paper sx={{ p: 4, textAlign: 'center', background: 'linear-gradient(135deg, #181c24 0%, #232526 100%)', borderRadius: 4, boxShadow: '0 6px 24px 0 #00eaff22' }}>
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', mb: 2 }}>
        <img src="/brain-chip.png" alt="AI Brain Chip" style={{ width: 90, height: 90, borderRadius: 18, boxShadow: '0 0 32px 0 #00eaff44', background: '#181c24' }} />
      </Box>
      <Typography variant="h5" sx={{ color: '#00eaff', fontWeight: 700, mt: 2, mb: 1, letterSpacing: 1 }}>
        모델 가중치 로드 완료
      </Typography>
      <Typography variant="subtitle1" sx={{ color: '#b2ebf2', fontWeight: 500 }}>
        모델명: <b>{modelId}</b>
      </Typography>
      <Typography variant="body2" sx={{ color: '#7cffcb', mt: 2 }}>
        AI가 활성화되었습니다.
      </Typography>
    </Paper>
  );
};

export default ModelLoadedDisplay; 