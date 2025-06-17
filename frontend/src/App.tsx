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
import ProgressDisplay from './components/ProgressDisplay';
import { useWebSocket } from './hooks/useWebSocket';
import './global.css';
import { SelectChangeEvent } from '@mui/material';
import PsychologyIcon from '@mui/icons-material/Psychology';
import ConfusionMatrixDisplay from './components/ConfusionMatrixDisplay';
import ConfusionMatrixGraph from './components/ConfusionMatrixGraph';
import CumulativeAccuracyGraph from './components/CumulativeAccuracyGraph';
import MetricsBarChart from './components/MetricsBarChart';
import VideoPlayer from './components/VideoPlayer';

const theme = createTheme({
    palette: {
        mode: 'light', primary: { main: '#1976d2' }, secondary: { main: '#00bfae' },
        background: { default: '#f5f7fa', paper: '#fff' }, text: { primary: '#222', secondary: '#555' },
    },
});

interface UploadedFile { name: string; size: number; duration?: number; }

interface AnnotationSegment {
    start_frame: number;
    end_frame: number;
    label: string;
}

interface VideoAnnotation {
    AR?: {
        label: string;
    };
    AL?: AnnotationSegment[];
}

interface AnnotationData {
    [videoName: string]: VideoAnnotation;
}

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
    const [resultVideos, setResultVideos] = useState<string[]>([]);
    const [selectedVideo, setSelectedVideo] = useState<string>('');
    const [totalFilesToUpload, setTotalFilesToUpload] = useState(0);
    const [uploadedFileCount, setUploadedFileCount] = useState(0);
    const [selectedUploadedFileName, setSelectedUploadedFileName] = useState<string>('');
    const [currentUploadSessionFiles, setCurrentUploadSessionFiles] = useState<string[]>([]);
    const [isInferring, setIsInferring] = useState<boolean>(false);
    const [realtimeOverlayFrame, setRealtimeOverlayFrame] = useState<string | null>(null);
    const [isAnalysisVideoSelectOpen, setIsAnalysisVideoSelectOpen] = useState(false);
    const [inferenceMode, setInferenceMode] = useState<'default' | 'AR' | 'AL'>('default');
    const [annotationData, setAnnotationData] = useState<AnnotationData>({});
    const [cumulativeAccuracyHistory, setCumulativeAccuracyHistory] = useState<{ processed_clips: number; accuracy: number; }[]>([]);
    const [metricsHistory, setMetricsHistory] = useState<any[]>([]);

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
        let ws: WebSocket | null = null;
        ws = new WebSocket('ws://localhost:10000/ws/realtime_overlay');
        ws.onopen = () => {
            console.log('실시간 오버레이 WebSocket 연결됨');
        };
        ws.onmessage = (event) => {
            setRealtimeOverlayFrame(`data:image/jpeg;base64,${event.data}`);
        };
        ws.onerror = (error) => {
            console.error('실시간 오버레이 WebSocket 오류:', error);
        };
        ws.onclose = () => {
            console.log('실시간 오버레이 WebSocket 연결 해제됨');
            setRealtimeOverlayFrame(null);
        };
        return () => {
            if (ws) {
                ws.close();
            }
        };
    }, []);

    useEffect(() => {
        if (inferenceState) {
            setIsInferring(inferenceState!.is_inferencing);
            if (inferenceState.cumulative_accuracy !== undefined && inferenceState.processed_videos !== undefined) {
                setCumulativeAccuracyHistory(prev => [
                    ...prev, 
                    { processed_clips: inferenceState.processed_videos, accuracy: inferenceState.cumulative_accuracy }
                ]);
            }
            if (inferenceState.metrics !== undefined) {
                setMetricsHistory(prev => [...prev, inferenceState.metrics]);
            }
        }
    }, [inferenceState]);

    useEffect(() => {
        if (videoRef.current) {
            videoRef.current.addEventListener('loadedmetadata', () => {
                console.log('비디오 메타데이터 로드 완료');
            });
            videoRef.current.addEventListener('canplay', () => {
                console.log('비디오 재생 가능');
            });
        }
    }, []);

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
            const data = await res.json();
            setModelId(id); setModelStatus('loaded');
            if (data.class_labels) {
                setClassLabels(data.class_labels);
            }
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
        setCurrentUploadSessionFiles([]);

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
                setCurrentUploadSessionFiles(prev => [...prev, data.files[0].name]);
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
        if (selectedUploadedFileName === fileName) {
            setSelectedUploadedFileName('');
        }
    };

    const handleRemoveAllFiles = async () => {
        await fetch('http://localhost:10000/uploads', { method: 'DELETE' });
        setUploadedFiles([]);
        setSelectedUploadedFileName('');
    };

    const handleFileSelectChange = (event: SelectChangeEvent<string>) => {
        setSelectedUploadedFileName(event.target.value as string);
    };

    const handleStartInference = () => {
        fetch('http://localhost:10000/infer', {
            method: 'POST', 
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                interval: frameInterval, 
                infer_period: inferPeriod, 
                batch: batchFrames,
                inference_mode: inferenceMode,
                annotation_data: annotationData
            }),
        });
    };

    const handleStopInference = () => {
        fetch('http://localhost:10000/stop_infer', { method: 'POST' });
    };

    const handleAnnotationUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                try {
                    const text = event.target?.result as string;
                    let parsedData: AnnotationData = {};
                    
                    if (file.name.endsWith('.txt')) {
                        const lines = text.split('\n').filter(line => line.trim());
                        
                        lines.forEach(line => {
                            const parts = line.split(' ');
                            const videoName = parts[0];
                            const label = parts[1];
                            
                            if (!parsedData[videoName]) {
                                parsedData[videoName] = {};
                            }
                            
                            if (inferenceMode === 'AR') {
                                if (parts.length !== 2) {
                                    throw new Error(`AR 모드에서는 비디오 이름과 레이블만 필요합니다: ${line}`);
                                }
                                parsedData[videoName]['AR'] = {
                                    label: label.trim()
                                };
                            }
                            else if (inferenceMode === 'AL') {
                                if (!parsedData[videoName]['AL']) {
                                    parsedData[videoName]['AL'] = [];
                                }
                                
                                if (label.toLowerCase().includes('non')) {
                                    if (parts.length !== 2) {
                                        throw new Error(`NonFight 레이블에는 프레임 번호가 필요하지 않습니다: ${line}`);
                                    }
                                    parsedData[videoName]['AL'] = [];
                                } else {
                                    if (parts.length < 4) {
                                        throw new Error(`Fight 레이블에는 최소 하나의 시작/종료 프레임 쌍이 필요합니다: ${line}`);
                                    }
                                    for (let i = 2; i < parts.length; i += 2) {
                                        if (i + 1 < parts.length) {
                                            parsedData[videoName]['AL']?.push({
                                                start_frame: parseInt(parts[i]),
                                                end_frame: parseInt(parts[i + 1]),
                                                label: label.trim()
                                            });
                                        }
                                    }
                                }
                            }
                        });
                    } else {
                        parsedData = JSON.parse(text);
                    }
                    
                    setAnnotationData(parsedData);
                    alert('어노테이션 파일이 성공적으로 로드되었습니다.');
                } catch (error: any) {
                    console.error("어노테이션 파일 파싱 오류:", error);
                    alert(`어노테이션 파일 파싱에 실패했습니다: ${error.message}`);
                    setAnnotationData({});
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
    
    const handleOpenAnalysisVideoSelect = async () => {
        try {
            const res = await fetch('http://localhost:10000/results/videos');
            if (!res.ok) throw new Error('결과 비디오 로딩 실패');
            const data = await res.json();
            if (data.videos && Array.isArray(data.videos)) {
                setResultVideos(data.videos);
                setIsAnalysisVideoSelectOpen(true);
            } else {
                alert('분석된 비디오가 없습니다.');
            }
        } catch (err) {
            console.error("결과 비디오 로딩 실패:", err);
            alert("분석 비디오 로딩에 실패했습니다. 백엔드 서버 로그를 확인해 주세요.");
        }
    };

    const handleCloseAnalysisVideoSelect = () => {
        setIsAnalysisVideoSelectOpen(false);
    };

    const handleSelectAnalysisVideo = (videoUrl: string) => {
        setSelectedVideo(videoUrl);
        setIsAnalysisVideoSelectOpen(false);
    };

    const selectedVideoUrl = selectedVideo ? `http://localhost:10000/video/${selectedVideo.replace('_overlay.mp4', '')}/overlay` : '';

    return (
        <ThemeProvider theme={theme}>
            <CssBaseline />
            <Container maxWidth={false} sx={{ p: '16px !important', height: '100vh', display: 'flex', flexDirection: 'column' }}>
                <Grid container spacing={1} sx={{ flexGrow: 1, minHeight: 0 }}>
                    <Grid item xs={12} md={3} sx={{ display: 'flex', flexDirection: 'column', flexGrow: 1, minHeight: 0 }}>
                        <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column', flexGrow: 1, flexShrink: 0, mb: 1 }}>
                            {renderModelLoader()}
                        </Paper>
                        <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column', justifyContent: 'center', flexGrow: 1, flexShrink: 0, overflowY: 'auto', mb: 1 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                                <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>Video Upload</Typography>
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
                                                maxHeight: 200,
                                                overflowY: 'auto',
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
                        <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column', flexGrow: 1, flexShrink: 0, overflowY: 'auto', mb: 1}}>
                            <Typography variant="subtitle1" sx={{ mb: 3, fontWeight: 'bold'}}>Inference Setting</Typography>
                            <Grid container spacing={1} sx={{ mb: 1 }}>
                                <Grid item xs={4}>
                                    <TextField
                                        label="배치 구간(Frames)"
                                        type="number"
                                        value={frameInterval}
                                        onChange={(e) => setFrameInterval(Number(e.target.value))}
                                        size="small"
                                        fullWidth sx={{ fontSize: '0.7rem' }}
                                    />
                                </Grid>
                                <Grid item xs={4}>
                                    <TextField
                                        label="추론 주기(Frames)"
                                        type="number"
                                        value={inferPeriod}
                                        onChange={(e) => setInferPeriod(Number(e.target.value))}
                                        size="small"
                                        fullWidth sx={{ fontSize: '0.7rem' }}
                                    />
                                </Grid>
                                <Grid item xs={4}>
                                    <TextField
                                        label="추출 프레임(Batch)"
                                        type="number"
                                        value={batchFrames}
                                        onChange={(e) => setBatchFrames(Number(e.target.value))}
                                        size="small"
                                        fullWidth sx={{ fontSize: '0.7rem' }}
                                    />
                                </Grid>
                            </Grid>
                            <Grid container spacing={1} sx={{ mb: 1 }}>
                                <Grid item xs={4}>
                                    <Button 
                                        variant={inferenceMode === 'AR' ? 'contained' : 'outlined'} 
                                        size="small" 
                                        onClick={() => setInferenceMode('AR')}
                                        fullWidth
                                    >
                                        AR
                                    </Button>
                                </Grid>
                                <Grid item xs={4}>
                                    <Button 
                                        variant={inferenceMode === 'AL' ? 'contained' : 'outlined'} 
                                        size="small" 
                                        onClick={() => setInferenceMode('AL')}
                                        fullWidth
                                    >
                                        AL
                                    </Button>
                                </Grid>
                                <Grid item xs={4}>
                                    <Button variant="outlined" component="label" size="small" fullWidth sx={{ fontSize: '0.8rem' }}>
                                       Annotation 
                                        <input type="file" hidden accept=".json,.txt" onChange={handleAnnotationUpload} />
                                    </Button>
                                </Grid>
                            </Grid>
                            <Button
                                variant={isInferring ? 'contained' : 'contained'}
                                onClick={isInferring ? handleStopInference : handleStartInference}
                                disabled={!modelId || uploadedFiles.length === 0}
                                color={isInferring ? 'error' : 'primary'}
                                fullWidth
                                sx={{ mt: 0, mb: 1 }}
                            >
                                {isInferring ? 'STOP' : 'RUN'}
                            </Button>
                            <Box sx={{ mt: 0.5 }}>
                                {classLabels.length > 0 && <Typography variant="caption" color="text.secondary">{classLabels.length}개 클래스</Typography>}
                                {videoDuration > 0 && <Typography variant="caption" color="text.secondary" sx={{ ml: 2 }}>비디오 길이: {videoDuration}초</Typography>}
                            </Box>
                            <Box sx={{ mt: 0.5 }}>
                                <ProgressDisplay
                                    isInferencing={inferenceState.is_inferencing}
                                    currentVideo={inferenceState.current_video}
                                    currentProgress={inferenceState.current_progress}
                                    totalVideos={inferenceState.total_videos}
                                    processedVideos={inferenceState.processed_videos}
                                />
                            </Box>
                        </Paper>
                        <SystemInfo sx={{ p: 2, display: 'flex', flexDirection: 'column', flexGrow: 1, flexShrink: 0, overflowY: 'auto'}} />
                    </Grid>
                    <Grid item xs={12} md={6} sx={{ display: 'flex', flexDirection: 'column', flexGrow: 1, minHeight: 0 }}>
                        {/* 상단 스트림 영역 - flex: 3으로 설정하여 30% 높이 */}
                        <Box sx={{ flex: 3, display: 'flex', minHeight: 0, mb: 2 }}>
                            <Grid container sx={{ flexGrow: 1 }}>
                                <Grid item xs={12} md={6} sx={{ display: 'flex', flexDirection: 'column' }}>
                                    <Paper sx={{ 
                                        display: 'flex', 
                                        flexDirection: 'column', 
                                        bgcolor: 'black', 
                                        borderRadius: 1, 
                                        overflow: 'hidden', 
                                        flexGrow: 1, 
                                        marginRight: '8px', 
                                        minHeight: 0,
                                        p: 2
                                    }}>
                                        <Box sx={{ 
                                            display: 'flex', 
                                            justifyContent: 'center', 
                                            alignItems: 'center', 
                                            flexShrink: 0,
                                            mb: 1
                                        }}>
                                            <Typography variant="subtitle1" color="white">
                                                실시간 추론 영상 스트림
                                            </Typography>
                                        </Box>
                                        <Box sx={{ 
                                            flex: 1, 
                                            width: '100%', 
                                            display: 'flex', 
                                            justifyContent: 'center', 
                                            alignItems: 'center', 
                                            minHeight: 0,
                                            position: 'relative'
                                        }}>
                                            {realtimeOverlayFrame ? (
                                                <img 
                                                    src={realtimeOverlayFrame} 
                                                    alt="Realtime Overlay" 
                                                    style={{ 
                                                        maxWidth: '100%', 
                                                        maxHeight: '100%', 
                                                        width: 'auto',
                                                        height: 'auto',
                                                        objectFit: 'contain'
                                                    }} 
                                                />
                                            ) : (
                                                <Box sx={{ 
                                                    display: 'flex', 
                                                    justifyContent: 'center', 
                                                    alignItems: 'center', 
                                                    color: 'white' 
                                                }}>
                                                    <Typography>실시간 추론 스트림 로딩...</Typography>
                                                </Box>
                                            )}
                                        </Box>
                                    </Paper>
                                </Grid>
                                <Grid item xs={12} md={6} sx={{ display: 'flex', flexDirection: 'column' }}>
                                    <Paper sx={{ 
                                        display: 'flex', 
                                        flexDirection: 'column', 
                                        bgcolor: 'black', 
                                        borderRadius: 1, 
                                        overflow: 'hidden', 
                                        flexGrow: 1, 
                                        marginLeft: '8px', 
                                        minHeight: 0,
                                        p: 2
                                    }}>
                                        <Box sx={{ 
                                            display: 'flex', 
                                            justifyContent: 'center', 
                                            alignItems: 'center', 
                                            flexShrink: 0,
                                            mb: 1
                                        }}>
                                            <Typography variant="subtitle1" color="white">
                                                추론 결과 분석
                                            </Typography>
                                        </Box>
                                        <Box sx={{ 
                                            flex: 1, 
                                            width: '100%', 
                                            display: 'flex', 
                                            justifyContent: 'center', 
                                            alignItems: 'center', 
                                            minHeight: 0
                                        }}>
                                            {selectedVideo ? (
                                                <VideoPlayer videoUrl={selectedVideoUrl} />
                                            ) : (
                                                <Box sx={{ 
                                                    display: 'flex', 
                                                    flexDirection: 'column', 
                                                    justifyContent: 'center', 
                                                    alignItems: 'center', 
                                                    color: 'white',
                                                    textAlign: 'center'
                                                }}>
                                                    <Typography variant="body2" sx={{ mb: 2 }}>
                                                        분석 비디오를 선택하세요
                                                    </Typography>
                                                    <Button 
                                                        variant="contained" 
                                                        size="small" 
                                                        onClick={handleOpenAnalysisVideoSelect}
                                                    >
                                                        추론 결과 영상 열기
                                                    </Button>
                                                </Box>
                                            )}
                                        </Box>
                                    </Paper>
                                </Grid>
                            </Grid>
                        </Box>
                        
                        {/* 하단 메트릭스 영역 - flex: 7로 설정하여 70% 높이 */}
                        <Box sx={{ flex: 7, minHeight: 0 }}>
                            <Paper sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                                <Grid container spacing={2} sx={{ flexGrow: 1, height: '100%' }}>
                                    {/* 1. 혼동 행렬 그래프 (왼쪽 절반) */}
                                    <Grid item xs={12} md={6} sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                                        <ConfusionMatrixDisplay metrics={metricsHistory[metricsHistory.length - 1]} />
                                    </Grid>

                                    {/* 오른쪽 절반 - 성능 지표 및 누적 정확도 그래프 */}
                                    <Grid item xs={12} md={6} sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                                        <MetricsBarChart metrics={metricsHistory[metricsHistory.length - 1]} />
                                        <CumulativeAccuracyGraph cumulativeAccuracyHistory={cumulativeAccuracyHistory} />
                                    </Grid>
                                </Grid>
                            </Paper>
                        </Box>
                    </Grid>

                    <Grid item xs={12} md={3} sx={{ display: 'flex', flexDirection: 'column', flexGrow: 1, minHeight: 0, height: '100%' }}>
                        {/* Inference Results Table - 70% height */}
                        <Box sx={{ height: '60%', display: 'flex', flexDirection: 'column', mb: 2, minHeight: 0 }}>
                            <InferenceResultTable events={inferenceState.events} classLabels={classLabels} />
                        </Box>
                        {/* Confusion Matrix Graph - 30% height */}
                        <Box sx={{ height: '40%', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', minHeight: 0 }}>
                            <ConfusionMatrixGraph metrics={inferenceState.metrics} />
                        </Box>
                    </Grid>
                </Grid>
                <Dialog open={isAnalysisVideoSelectOpen} onClose={handleCloseAnalysisVideoSelect} maxWidth="sm" fullWidth>
                    <DialogTitle>분석 비디오 선택</DialogTitle>
                    <DialogContent>
                        {resultVideos.length === 0 ? (
                            <Typography>분석된 비디오가 없습니다.</Typography>
                        ) : (
                            <List>
                                {resultVideos.map((video, index) => (
                                    <ListItem button key={index} onClick={() => handleSelectAnalysisVideo(video)} selected={selectedVideo === video}>
                                        <ListItemText primary={video.substring(video.lastIndexOf('/') + 1)} />
                                    </ListItem>
                                ))}
                            </List>
                        )}
                    </DialogContent>
                    <DialogActions>
                        <Button onClick={handleCloseAnalysisVideoSelect}>취소</Button>
                        <Button onClick={() => handleSelectAnalysisVideo(selectedVideo)} color="primary" disabled={!selectedVideo}>재생</Button>
                    </DialogActions>
                </Dialog>
            </Container>
        </ThemeProvider>
    );
}

export default App;
