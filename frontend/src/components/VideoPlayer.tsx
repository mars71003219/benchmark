import React from 'react';
import ReactPlayer from 'react-player';
import { Box, Typography } from '@mui/material';
import VideocamIcon from '@mui/icons-material/Videocam';

interface VideoPlayerProps {
  videoUrl: string | null;
}

const VideoPlayer: React.FC<VideoPlayerProps> = ({ videoUrl }) => {
  return (
    <Box sx={{ 
      width: '100%', 
      height: '100%', 
      display: 'flex', 
      flexDirection: 'column',
      minHeight: 0 
    }}>
      <Typography 
        variant="subtitle2" 
        gutterBottom 
        sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: 1,
          color: 'white',
          flexShrink: 0,
          mb: 1
        }}
      >
        <VideocamIcon fontSize="small" />
        비디오 플레이어
      </Typography>

      <Box
        sx={{
          flex: 1,
          position: 'relative',
          width: '100%',
          minHeight: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}
      >
        {videoUrl ? (
          <ReactPlayer
            url={videoUrl}
            width="100%"
            height="100%"
            controls={true}
            style={{
              maxWidth: '100%',
              maxHeight: '100%'
            }}
            config={{
              file: {
                attributes: {
                  style: {
                    width: '100%',
                    height: '100%',
                    objectFit: 'contain'
                  }
                }
              }
            }}
          />
        ) : (
          <Typography color="white" variant="body2">
            재생할 비디오를 선택해주세요
          </Typography>
        )}
      </Box>
    </Box>
  );
};

export default VideoPlayer;