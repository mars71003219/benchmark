// /frontend/src/App.tsx

import React, { useState, useEffect, useRef } from 'react';
import {
    Container, Box, Typography, Paper, ThemeProvider, createTheme, CssBaseline, Button,
    TextField, Grid, List, ListItem, ListItemText, ListItemSecondaryAction, IconButton,
    Stack, CircularProgress, LinearProgress, Dialog, DialogTitle, DialogContent, DialogActions,
    FormControl, InputLabel, Select, MenuItem
} from '@mui/material';
import FolderOpenIcon from '@mui/icons-material/FolderOpen';
import DeleteIcon from '@mui/icons-material/Delete';
import DownloadIcon from '@mui/icons-material/Download';
import SystemInfo from './components/SystemInfo';
import VideoResultViewer from './components/VideoResultViewer';
import InferenceResultTable from './components/InferenceResultTable';
import ModelSelector from './components/ModelSelector';
import ModelLoadingAnimation from './components/ModelLoadingAnimation';
import VideoInferenceChart from './components/VideoInferenceChart';
import ProgressDisplay from './components/ProgressDisplay';
import { useWebSocket } from './hooks/useWebSocket';
import Hls from 'hls.js';
import './global.css';
import { SelectChangeEvent } from '@mui/material';

const theme = createTheme({
    palette: {
        mode: 'light', primary: { main: '#1976d2' }, secondary: { main: '#00bfae' },
        background: { default: '#f5f7fa', paper: '#fff' }, text: { primary: '#222', secondary: '#555' },
    },
});

interface UploadedFile { name: string; size: number; duration?: number; }

function App() {
    const [modelId, setModelId] = useState('');
    const [modelStatus, setModelStatus] = useState<'none' | 'loading' | 'loaded'>('none');
    const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
    const [frameInterval, setFrameInterval] = useState(90);
    const [inferPeriod, setInferPeriod] = useState(30);
    const [batchFrames, setBatchFrames] = useState(16);
    const inferenceState = useWebSocket('ws://localhost:10000/ws');
    const [settingsOpen, setSettingsOpen] = useState(false);
    const [classLabels, setClassLabels] = useState<string[]>([]);
    const [videoDuration, setVideoDuration] = useState(0);
    const videoRef = useRef<HTMLVideoElement>(null);
    const [streamMode, setStreamMode] = useState<'realtime' | 'analysis'>('realtime');
    const [resultVideos, setResultVideos] = useState<string[]>([]);
    const [selectedVideo, setSelectedVideo] = useState<string>('');
    const [isPlayerVisible, setPlayerVisible] = useState(true);
    const [totalFilesToUpload, setTotalFilesToUpload] = useState(0);
    const [uploadedFileCount, setUploadedFileCount] = useState(0);
    const [selectedUploadedFileName, setSelectedUploadedFileName] = useState<string>('');
    const [currentUploadSessionFiles, setCurrentUploadSessionFiles] = useState<string[]>([]);

    useEffect(() => {
        const fetchFiles = async () => {
            try {
                const res = await fetch('http://localhost:10000/uploads');
                if (!res.ok) throw new Error('업로드 파일 로딩 실패');
                const data = await res.json();
                if (Array.isArray(data.files)) {
                    setUploadedFiles(data.files);
                    if (data.files.length > 0) {
                        setSelectedUploadedFileName(data.files[0].name);
                    }
                }
            } catch (error) {
                console.error("파일 로드 오류:", error);
            }
        };
        fetchFiles();
    }, []);

    useEffect(() => {
        if (uploadedFiles.length === 0) {
            setSelectedUploadedFileName('');
        } else if (!selectedUploadedFileName || !uploadedFiles.some(f => f.name === selectedUploadedFileName)) {
            setSelectedUploadedFileName(uploadedFiles[0].name);
        }
    }, [uploadedFiles, selectedUploadedFileName]);

    useEffect(() => {
        if (videoRef.current && Hls.isSupported()) {
            const hls = new Hls();
            hls.loadSource('http://localhost:8554/overlay.m3u8');
            hls.attachMedia(videoRef.current);
            hls.on(Hls.Events.MANIFEST_PARSED, () => {
                if (videoRef.current) {
                    videoRef.current.play();
                }
            });
        }
    }, []);

    useEffect(() => {
        if (streamMode === 'analysis') {
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
    }, [streamMode]);

    if (!inferenceState) {
        return (
             <ThemeProvider theme={theme}>
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
                    <CircularProgress /> <Typography sx={{ ml: 2 }}>서버와 연결 중...</Typography>
                </Box>
            </ThemeProvider>
        );
    }

    const handleLoadModel = async (id: string) => {
        setModelStatus('loading');
        try {
            const res = await fetch('http://localhost:10000/model', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_id: id }),
            });
            if (!res.ok) throw new Error('모델 로딩 실패');
            setModelId(id); setModelStatus('loaded');
        } catch (error) {
            alert("모델 로딩에 실패했습니다. 백엔드 서버 로그를 확인해 주세요.");
            setModelStatus('none');
        }
    };
    
    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (!files || files.length === 0) return;

        setTotalFilesToUpload(files.length);
        setUploadedFileCount(0);
        setCurrentUploadSessionFiles([]); // Reset for new upload session

        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const formData = new FormData();
            formData.append('files', file);
            formData.append('paths', file.webkitRelativePath || file.name);
            
            try {
                const res = await fetch('http://localhost:10000/upload', { method: 'POST', body: formData });
                if (!res.ok) throw new Error(`파일 업로드 실패: ${file.name}`);
                const data = await res.json();
                
                setUploadedFiles(prev => {
                    const newFiles = [...prev, ...data.files];
                    return newFiles.filter((v, idx, a) => a.findIndex(t => (t.name === v.name)) === idx);
                });
                setUploadedFileCount(prev => prev + 1);
                setCurrentUploadSessionFiles(prev => [...prev, data.files[0].name]); // Add to current session files
            } catch (error) {
                console.error(error);
                alert(error); 
            }
        }
        e.target.value = ''; 
        setTotalFilesToUpload(0);
        setUploadedFileCount(0);
    };

    const handleRemoveFile = async (fileName: string) => {
        await fetch(`http://localhost:10000/upload/${encodeURIComponent(fileName)}`, { method: 'DELETE' });
        setUploadedFiles(prev => prev.filter(f => f.name !== fileName));
        // If the deleted file was the selected one, clear selection
        if (selectedUploadedFileName === fileName) {
            setSelectedUploadedFileName('');
        }
    };

    const handleRemoveAllFiles = async () => {
        await fetch('http://localhost:10000/uploads', { method: 'DELETE' });
        setUploadedFiles([]);
        setSelectedUploadedFileName(''); // Clear selection after all files removed
    };

    const handleFileSelectChange = (event: SelectChangeEvent<string>) => {
        setSelectedUploadedFileName(event.target.value as string);
    };

    const handleStartInference = () => {
        fetch('http://localhost:10000/infer', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ interval: frameInterval, infer_period: inferPeriod, batch: batchFrames }),
        });
    };

    const handleStopInference = () => {
        fetch('http://localhost:10000/stop_infer', { method: 'POST' });
    };

    const handleClassTxtUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                if (event.target) {
                    const text = event.target.result as string;
                    setClassLabels(text.split('\n').map(line => line.trim()));
                }
            };
            reader.readAsText(file);
        }
    };

    const handleAnnoTxtUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                if (event.target) {
                    const text = event.target.result as string;
                    setVideoDuration(Number(text.split('\n')[0].trim()));
                }
            };
            reader.readAsText(file);
        }
    };

    const renderModelLoader = () => {
        switch (modelStatus) {
            case 'loading': return <ModelLoadingAnimation isLoading={true} />;
            case 'loaded': return (
                <Box sx={{ textAlign: 'center', width: '100%' }}>
                    <Typography variant="subtitle1" color="primary" sx={{fontWeight: 'bold', wordBreak: 'break-all'}}>{modelId}</Typography>
                    <Typography variant="body2" color="text.secondary">모델 로드 완료</Typography>
                    <Button variant="outlined" size="small" onClick={() => { setModelId(''); setModelStatus('none'); }} sx={{mt: 1}}>해제</Button>
                </Box>
            );
            default: return <ModelSelector onSubmit={handleLoadModel} />;
        }
    };
    
    const isAnalysisComplete = inferenceState.events.some(ev => ev.type === 'complete');
    
    return (
        <ThemeProvider theme={theme}>
            <CssBaseline />
            <Container maxWidth={false} sx={{ p: '16px !important', height: '100vh', display: 'flex', flexDirection: 'column' }}>
                <Grid container spacing={2} sx={{ flexGrow: 1, minHeight: 0, height: '100%' }}>
                    {/* 좌측: 모델/업로드/설정/시스템 */}
                    <Grid item xs={12} md={3} sx={{ display: 'flex', flexDirection: 'column', gap: 2, flexGrow: 1, minHeight: 0 }}>
                        {/* 모델 로드 */}
                        <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', flexGrow: 0, flexShrink: 0, height: 100 }}>
                            {renderModelLoader()}
                        </Paper>
                        {/* 비디오 업로드 */}
                        <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column', justifyContent: 'center', flexGrow: 0, flexShrink: 0, overflowY: 'auto', height: 200 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                                <Typography variant="subtitle1">비디오 테스트셋</Typography>
                                <Box>
                                </Box>
                            </Box>
                            {totalFilesToUpload > 0 && (
                                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                                    업로드 중: {uploadedFileCount} / {totalFilesToUpload} ({((uploadedFileCount / totalFilesToUpload) * 100).toFixed(0)}%)
                                </Typography>
                            )}
                            <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
                                <Button variant="outlined" component="label" startIcon={<FolderOpenIcon />} size="small" sx={{ flex: 1 }}>파일 열기<input type="file" hidden multiple onChange={handleFileUpload} /></Button>
                                <Button variant="outlined" component="label" startIcon={<FolderOpenIcon />} size="small" sx={{ flex: 1 }}>폴더 열기<input type="file" hidden multiple webkitdirectory="" onChange={handleFileUpload} /></Button>
                                <Button variant="text" color="error" size="small" onClick={handleRemoveAllFiles}>전체 삭제</Button>
                            </Box>
                            
                            <FormControl fullWidth size="small" sx={{ mt: 1 }}>
                                <InputLabel id="uploaded-video-select-label">업로드된 비디오</InputLabel>
                                <Select
                                    labelId="uploaded-video-select-label"
                                    value={selectedUploadedFileName}
                                    label="업로드된 비디오"
                                    onChange={handleFileSelectChange}
                                    MenuProps={{
                                        PaperProps: {
                                            sx: {
                                                maxHeight: 200, // Max height for scroll
                                                overflowY: 'auto', // Enable scroll
                                            },
                                        },
                                    }}
                                >
                                    {uploadedFiles.map((f) => (
                                        <MenuItem key={f.name} value={f.name}>
                                            <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                                                <Typography
                                                    component="span"
                                                    variant="body2"
                                                    sx={{
                                                        fontWeight: 'bold',
                                                        flexShrink: 1,
                                                        minWidth: 0,
                                                        overflow: 'hidden',
                                                        textOverflow: 'ellipsis',
                                                        whiteSpace: 'nowrap'
                                                    }}
                                                >
                                                    {f.name}
                                                </Typography>
                                                <Typography
                                                    component="span"
                                                    variant="body2"
                                                    sx={{ ml: 2, flexShrink: 0 }}
                                                >
                                                    영상시간: {f.duration ? f.duration.toFixed(2) + 's' : '로딩 중...'}
                                                </Typography>
                                            </Box>
                                        </MenuItem>
                                    ))}
                                </Select>
                            </FormControl>

                        </Paper>
                        {/* 추론 설정 */}
                        <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column', justifyContent: 'center', flexGrow: 0, flexShrink: 0, overflowY: 'auto', height: 300 }}>
                            <Typography variant="h6" sx={{ mb: 1 }}>추론 설정</Typography>
                            {/* 업로드 버튼 2개를 한 줄에 나란히 */}
                            <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                                <Button variant="outlined" component="label" size="small" sx={{ flex: 1, minWidth: 0 }}>Class txt 업로드<input type="file" hidden accept=".txt" onChange={handleClassTxtUpload} /></Button>
                                <Button variant="outlined" component="label" size="small" sx={{ flex: 1, minWidth: 0 }}>Annotation txt 업로드<input type="file" hidden accept=".txt" onChange={handleAnnoTxtUpload} /></Button>
                            </Box>
                            <TextField label="샘플링 구간(Frames)" type="number" value={frameInterval} onChange={e => setFrameInterval(Number(e.target.value))} size="small" fullWidth sx={{ mb: 1 }} />
                            <TextField label="추론 주기(Frames)" type="number" value={inferPeriod} onChange={e => setInferPeriod(Number(e.target.value))} size="small" fullWidth sx={{ mb: 1 }} />
                            <TextField label="샘플링 프레임(Batch)" type="number" value={batchFrames} onChange={e => setBatchFrames(Number(e.target.value))} size="small" fullWidth sx={{ mb: 1 }} />
                            <Button variant="contained" fullWidth sx={{ mt: 1, mb: 1 }} onClick={handleStartInference}>추론 실행</Button>
                            <Button variant="outlined" fullWidth size="small" color="error" onClick={handleStopInference}>추론 중지</Button>
                            <Box sx={{ mt: 1 }}>
                                {classLabels.length > 0 && <Typography variant="caption" color="text.secondary">{classLabels.length}개 클래스</Typography>}
                                {videoDuration > 0 && <Typography variant="caption" color="text.secondary" sx={{ ml: 2 }}>비디오 길이: {videoDuration}초</Typography>}
                            </Box>
                        </Paper>
                        {/* 시스템 정보 */}
                        <Box sx={{ flexGrow: 1, minHeight: 0 }}>
                            <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column', overflowY: 'auto', height: '100%' }}>
                                <SystemInfo />
                            </Paper>
                        </Box>
                    </Grid>
                    {/* 중앙: 실시간/분석 스트림 및 그래프 */}
                    <Grid item xs={12} md={6} sx={{ display: 'flex', flexDirection: 'column', gap: 2, height: '100%' }}>
                        {/* 실시간/분석 토글 및 비디오 재생 */}
                        <Paper sx={{ p: 2, flex: 1, display: 'flex', flexDirection: 'column' }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                                <Typography variant="subtitle1">{streamMode === 'realtime' ? '실시간 스트림' : '분석 결과'}</Typography>
                                <Box>
                                    <Button variant={streamMode === 'realtime' ? 'contained' : 'outlined'} size="small" sx={{ mr: 1 }} onClick={() => setStreamMode('realtime')}>실시간</Button>
                                    <Button variant={streamMode === 'analysis' ? 'contained' : 'outlined'} size="small" onClick={() => setStreamMode('analysis')}>분석</Button>
                                </Box>
                            </Box>
                            <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
                                {isPlayerVisible ? (
                                    <Box sx={{ position: 'relative', width: '100%', paddingTop: '56.25%', bgcolor: 'black', borderRadius: 1, overflow: 'hidden' }}>
                                        {streamMode === 'realtime' ? (
                                            <video ref={videoRef} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'contain' }} controls muted />
                                        ) : (
                                            selectedVideo ? (
                                                <video src={selectedVideo} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'contain' }} controls />
                                            ) : (
                                                <Box sx={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center', color: 'white' }}>
                                                    <Typography>분석 비디오를 선택하세요</Typography>
                                                </Box>
                                            )
                                        )}
                                    </Box>
                                ) : (
                                    <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', width: '100%' }}>
                                        <CircularProgress /> <Typography sx={{ ml: 2 }}>분석 비디오 로딩 중...</Typography>
                                    </Box>
                                )}
                            </Box>
                        </Paper>
                        {/* 비디오 추론 이벤트 그래프 */}
                        <Paper sx={{ p: 2, flex: 1, display: 'flex', flexDirection: 'column' }}>
                            <Typography variant="h6" gutterBottom>비디오 추론 이벤트 그래프</Typography>
                            {/* 실제 그래프 컴포넌트 */}
                            <Box sx={{ flexGrow: 1, display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 0 }}>
                                <VideoInferenceChart events={inferenceState.events} classLabels={classLabels} videoDuration={videoDuration} />
                            </Box>
                        </Paper>
                    </Grid>
                    {/* 우측: 실시간 추론 이벤트 및 진행률 */}
                    <Grid item xs={12} md={3} sx={{ display: 'flex', flexDirection: 'column', gap: 2, height: '100%', minHeight: 0 }}>
                        {/* 실시간 추론 이벤트 */}
                        <Paper sx={{ p: 2, flex: 1, display: 'flex', flexDirection: 'column' }}>
                            <Typography variant="h6" gutterBottom>실시간 추론 이벤트</Typography>
                            <InferenceResultTable events={inferenceState.events} classLabels={classLabels} />
                        </Paper>
                        {/* 단일 클립 진행률 / 전체 비디오 진행률 */}
                        <Paper sx={{ p: 2, flexShrink: 0 }}>
                            <ProgressDisplay
                                isInferencing={inferenceState.is_inferencing}
                                currentVideo={inferenceState.current_video}
                                currentProgress={inferenceState.current_progress}
                                totalVideos={inferenceState.total_videos}
                                processedVideos={inferenceState.processed_videos}
                            />
                        </Paper>
                    </Grid>
                </Grid>
            </Container>
        </ThemeProvider>
    );
}

export default App;
