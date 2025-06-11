import React, { useRef } from 'react';
import ReactPlayer from 'react-player';
import { Box, Typography, Paper } from '@mui/material';
import VideocamIcon from '@mui/icons-material/Videocam';

interface VideoPlayerProps {
  videoUrl: string | null;
}

const VideoPlayer: React.FC<VideoPlayerProps> = ({ videoUrl }) => {
  const playerRef = useRef<ReactPlayer>(null);

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
            ref={playerRef}
            url={videoUrl}
            width="100%"
            height="100%"
            controls
            style={{
              position: 'absolute',
              top: 0,
              left: 0
            }}
            config={{
              file: {
                forceVideo: true,
                attributes: {
                  controls: true,
                  preload: 'auto',
                  crossOrigin: 'anonymous'
                }
              }
            }}
            onReady={() => {
              if (playerRef.current) {
                const videoElement = playerRef.current.getInternalPlayer() as HTMLVideoElement;
                if (videoElement) {
                  videoElement.addEventListener('seeking', () => {
                    const currentTime = videoElement.currentTime;
                    const duration = videoElement.duration;
                    if (duration > 0) {
                      videoElement.currentTime = currentTime;
                    }
                  });
                }
              }
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