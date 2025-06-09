// frontend/src/components/VideoResultViewer.tsx

import React, { useState, useEffect } from 'react';
import {
    Box, Typography, Paper, ToggleButtonGroup, ToggleButton,
    Select, MenuItem, FormControl, InputLabel, IconButton
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import ReactPlayer from 'react-player';

const VideoResultViewer: React.FC = () => {
    const [mode, setMode] = useState<'realtime' | 'analysis'>('realtime');
    const [resultVideos, setResultVideos] = useState<string[]>([]);
    const [selectedVideo, setSelectedVideo] = useState<string>('');
    const [isPlayerVisible, setPlayerVisible] = useState(true);

    useEffect(() => {
        if (mode === 'analysis') {
            setPlayerVisible(false);
            fetch('http://localhost:10000/results/videos')
                .then(res => res.json())
                .then(data => {
                    if (data.videos && data.videos.length > 0) {
                        setResultVideos(data.videos);
                        setSelectedVideo(data.videos[0]);
                        setPlayerVisible(true);
                    }
                }).catch(err => console.error("결과 비디오 로딩 실패:", err));
        } else {
            setPlayerVisible(true);
        }
    }, [mode]);

    const handleModeChange = (evt: React.MouseEvent<HTMLElement>, newMode: 'realtime' | 'analysis') => {
        if (newMode) setMode(newMode);
    };

    const selectedVideoUrl = selectedVideo ? `http://localhost:10000/video/${selectedVideo.replace('_overlay.mp4', '')}/overlay` : '';

    return (
        <Paper sx={{ width: '100%', height: '100%', p: 2, display: 'flex', flexDirection: 'column' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1, flexShrink: 0 }}>
                <Typography variant="h6">
                    {mode === 'realtime' ? '실시간 스트림' : '분석 결과'}
                </Typography>
                <ToggleButtonGroup value={mode} exclusive onChange={handleModeChange} size="small">
                    <ToggleButton value="realtime">실시간</ToggleButton>
                    <ToggleButton value="analysis">분석</ToggleButton>
                </ToggleButtonGroup>
            </Box>

            <Box sx={{ flexGrow: 1, minHeight: 0, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                {isPlayerVisible ? (
                    <Box sx={{ position: 'relative', width: '100%', paddingTop: '56.25%', bgcolor: 'black', borderRadius: 1, overflow: 'hidden' }}>
                        {mode === 'realtime' ? (
                            <Box sx={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                <Typography sx={{ color: 'white' }}>RTSP 스트림 재생 영역</Typography>
                            </Box>
                        ) : (
                            resultVideos.length > 0 ?
                            <ReactPlayer url={selectedVideoUrl} width="100%" height="100%" controls playing style={{ position: 'absolute', top: 0, left: 0 }} />
                            : <Box sx={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}><Typography color="white">생성된 비디오 없음</Typography></Box>
                        )}
                    </Box>
                ) : (
                    <Box sx={{ flexGrow: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', bgcolor: 'grey.200', borderRadius: 1 }}>
                        <Typography color="text.secondary">표시할 분석 결과가 없습니다.</Typography>
                    </Box>
                )}
            </Box>

            {mode === 'analysis' && isPlayerVisible && resultVideos.length > 0 && (
                <Box sx={{ display: 'flex', alignItems: 'center', mt: 2, gap: 1, flexShrink: 0 }}>
                    <FormControl fullWidth size="small">
                        <InputLabel>결과 비디오 선택</InputLabel>
                        <Select value={selectedVideo} label="결과 비디오 선택" onChange={(e) => setSelectedVideo(e.target.value)}>
                            {resultVideos.map(v => <MenuItem key={v} value={v}>{v}</MenuItem>)}
                        </Select>
                    </FormControl>
                    <IconButton onClick={() => setPlayerVisible(false)} title="플레이어 닫기"><CloseIcon /></IconButton>
                </Box>
            )}
        </Paper>
    );
};

export default VideoResultViewer;
