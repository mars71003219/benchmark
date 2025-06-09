import React from 'react';
import { Box, Typography } from '@mui/material';
import { keyframes } from '@mui/system';
import { styled } from '@mui/material/styles';

const pulse = keyframes`
  0% {
    transform: scale(0.95);
    box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7);
  }
  
  70% {
    transform: scale(1);
    box-shadow: 0 0 0 10px rgba(76, 175, 80, 0);
  }
  
  100% {
    transform: scale(0.95);
    box-shadow: 0 0 0 0 rgba(76, 175, 80, 0);
  }
`;

const LoadingCircle = styled(Box)(({ theme }) => ({
  width: '60px',
  height: '60px',
  borderRadius: '50%',
  background: 'linear-gradient(45deg, #4caf50, #81c784)',
  animation: `${pulse} 2s infinite`,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  margin: '0 auto',
  position: 'relative',
  '&::after': {
    content: '""',
    position: 'absolute',
    width: '100%',
    height: '100%',
    borderRadius: '50%',
    border: '2px solid #4caf50',
    animation: `${pulse} 2s infinite`,
    animationDelay: '0.5s',
  }
}));

const Checkmark = styled(Box)(({ theme }) => ({
  width: '30px',
  height: '30px',
  position: 'relative',
  '&::before': {
    content: '""',
    position: 'absolute',
    width: '6px',
    height: '12px',
    border: 'solid white',
    borderWidth: '0 3px 3px 0',
    transform: 'rotate(45deg)',
    top: '2px',
    left: '10px',
  }
}));

interface ModelLoadingAnimationProps {
  isLoading: boolean;
}

const ModelLoadingAnimation: React.FC<ModelLoadingAnimationProps> = ({ isLoading }) => {
  return (
    <Box sx={{ textAlign: 'center', py: 2 }}>
      {isLoading ? (
        <>
          <LoadingCircle />
          <Typography variant="body2" sx={{ mt: 2, color: 'text.secondary' }}>
            모델 가중치 로딩 중...
          </Typography>
        </>
      ) : (
        <>
          <Box sx={{ 
            width: '60px', 
            height: '60px', 
            borderRadius: '50%', 
            background: '#4caf50',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto',
            animation: `${pulse} 2s infinite`
          }}>
            <Checkmark />
          </Box>
          <Typography variant="body2" sx={{ mt: 2, color: 'success.main' }}>
            모델 로드 완료
          </Typography>
        </>
      )}
    </Box>
  );
};

export default ModelLoadingAnimation; 