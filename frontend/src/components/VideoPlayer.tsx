import React from 'react';
import ReactPlayer from 'react-player';
import { Box, Typography, Paper } from '@mui/material';
import VideocamIcon from '@mui/icons-material/Videocam';

interface VideoPlayerProps {
  videoUrl: string | null;
}

const VideoPlayer: React.FC<VideoPlayerProps> = ({ videoUrl }) => {
  return (
    <Box>
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <VideocamIcon />
        비디오 플레이어
      </Typography>

      <Paper
        sx={{
          position: 'relative',
          paddingTop: '56.25%', // 16:9 비율
          bgcolor: 'background.paper',
          overflow: 'hidden'
        }}
      >
        {videoUrl ? (
          <ReactPlayer
            url={videoUrl}
            width="100%"
            height="100%"
            controls={true} // 기본 컨트롤 활성화
            style={{
              position: 'absolute',
              top: 0,
              left: 0
            }}
          />
        ) : (
          <Box
            sx={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              bgcolor: 'background.paper'
            }}
          >
            <Typography color="text.secondary">
              재생할 비디오를 선택해주세요
            </Typography>
          </Box>
        )}
      </Paper>
    </Box>
  );
};

export default VideoPlayer;