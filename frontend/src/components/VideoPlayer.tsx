import React from 'react';
import ReactPlayer from 'react-player';
import { Box, Typography } from '@mui/material';
import VideocamIcon from '@mui/icons-material/Videocam';

interface VideoPlayerProps {
  videoUrl: string | null;
}

const VideoPlayer: React.FC<VideoPlayerProps> = ({ videoUrl }) => {
  return (
    <Box
      sx={{
        width: '100%',
        position: 'relative',
        aspectRatio: '16/9', // 16:9 비율 고정
        bgcolor: 'black',
        borderRadius: 1,
        overflow: 'hidden',
        minHeight: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }}
    >
      {videoUrl ? (
        <Box sx={{
          position: 'absolute',
          top: 0, left: 0, right: 0, bottom: 0,
          width: '100%',
          height: '100%',
        }}>
          <ReactPlayer
            url={videoUrl}
            width="100%"
            height="100%"
            controls={true}
            style={{
              position: 'absolute',
              top: 0, left: 0, right: 0, bottom: 0,
              objectFit: 'contain',
              background: 'black'
            }}
            config={{
              file: {
                attributes: {
                  style: {
                    width: '100%',
                    height: '100%',
                    objectFit: 'contain',
                    background: 'black'
                  }
                }
              }
            }}
          />
        </Box>
      ) : (
        <Typography color="white" variant="body2">
          재생할 비디오를 선택해주세요
        </Typography>
      )}
    </Box>
  );
};

export default VideoPlayer;