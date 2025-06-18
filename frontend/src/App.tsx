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
import EventLog from './components/EventLog';

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

interface NasCopyProgress {
    is_copying: boolean;
    total_files: number;
    copied_files: number;
    current_file: string | null;
    errors: string[];
}

function App() {
    const [modelId, setModelId] = useState('');
    const [modelStatus, setModelStatus] = useState<'none' | 'loading' | 'loaded'>('none');
    const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
    const [frameInterval, setFrameInterval] = useState(90);
    const [inferPeriod, setInferPeriod] = useState(30);
    const [batchFrames, setBatchFrames] = useState(16);
    const inferenceState = useWebSocket('/ws');
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
    const [isPaused, setIsPaused] = useState(false);
    const [isAnnotationVideoMismatch, setIsAnnotationVideoMismatch] = useState(false);
    const [annotationAlertOpen, setAnnotationAlertOpen] = useState(false);
    const [annotationSuccessOpen, setAnnotationSuccessOpen] = useState(false);
    const [selectedResultFile, setSelectedResultFile] = useState<File | null>(null);
    
    // NAS 폴더 복사 관련 상태
    const [nasCopyProgress, setNasCopyProgress] = useState<NasCopyProgress>({
        is_copying: false,
        total_files: 0,
        copied_files: 0,
        current_file: null,
        errors: []
    });
    const [nasCopyProgressOpen, setNasCopyProgressOpen] = useState(false);
    
    // NAS 경로 설정 (백엔드에서 가져옴)
    const [nasBasePath, setNasBasePath] = useState('/home/hsnam');
    const [nasTargetPath, setNasTargetPath] = useState('/aivanas');

    // 1. 상태 추가
    const [nasInputOpen, setNasInputOpen] = useState(false);
    const [nasInputValue, setNasInputValue] = useState('');

    // 1. 복사 진행률 UI 표시 상태 관리
    const [showNasProgress, setShowNasProgress] = useState(false);

    useEffect(() => {
        // 1. 현재 로드된 모델 정보
        fetch('/current_model')
            .then(res => res.json())
            .then(data => {
                if (data.model_id) {
                    setModelId(data.model_id);
                    setModelStatus('loaded');
                } else {
                    setModelId('');
                    setModelStatus('none');
                }
            })
            .catch(() => {});
        
        // 2. 현재 추론 상태 정보
        fetch('/current_inference_state')
            .then(res => res.json())
            .then(data => {
                if (data && data.is_inferencing) {
                    setIsInferring(true);
                }
            })
            .catch(() => {});
        
        // 3. 현재 업로드된 파일 목록 새로고침
        const fetchFiles = async () => {
            try {
                const res = await fetch('/uploads');
                if (!res.ok) throw new Error('업로드 파일 로딩 실패');
                const data = await res.json();
                if (Array.isArray(data.files)) {
                    setUploadedFiles(data.files);
                    if (data.files.length > 0 && data.files[0] && data.files[0].name) {
                        setSelectedUploadedFileName(data.files[0].name);
                    } else {
                        setSelectedUploadedFileName('');
                    }
                }
            } catch (error) {
                console.error("파일 로드 오류:", error);
            }
        };
        fetchFiles();
        
        // 4. NAS 복사 진행 상태 확인
        fetch('/nas_copy_progress')
            .then(res => res.json())
            .then(data => {
                if (data && data.is_copying) {
                    setNasCopyProgress(data);
                    setNasCopyProgressOpen(true);
                }
            })
            .catch(() => {});
        
        // 5. NAS 경로 설정 가져오기
        fetch('/nas_paths')
            .then(res => res.json())
            .then(data => {
                setNasBasePath(data.base_path);
                setNasTargetPath(data.target_path);
                console.log(`NAS 경로 설정: ${data.base_path} -> ${data.target_path}`);
            })
            .catch(() => {
                console.log('NAS 경로 설정을 가져올 수 없어 기본값을 사용합니다.');
            });
    }, []);

    useEffect(() => {
        if (uploadedFiles.length === 0) {
            setSelectedUploadedFileName('');
        } else if (!selectedUploadedFileName || !uploadedFiles.some(f => f && f.name === selectedUploadedFileName)) {
            const validFiles = uploadedFiles.filter(f => f && f.name);
            if (validFiles.length > 0) {
                setSelectedUploadedFileName(validFiles[0].name);
            } else {
                setSelectedUploadedFileName('');
            }
        }
    }, [uploadedFiles, selectedUploadedFileName]);

    useEffect(() => {
        let ws: WebSocket | null = null;
        ws = new WebSocket('ws://192.168.190.4:10000/ws/realtime_overlay');
        ws.onopen = () => {
            console.log('실시간 오버레이 WebSocket 연결됨');
        };
        ws.onmessage = (event) => {
            if (event.data === "ping") {
                ws?.send("pong");
                return;
            }
            setRealtimeOverlayFrame(`data:image/jpeg;base64,${event.data}`);
        };
        ws.onerror = (error) => {
            console.error('실시간 오버레이 WebSocket 오류:', error);
        };
        ws.onclose = () => {
            console.log('실시간 오버레이 WebSocket 연결 해제됨');
            setRealtimeOverlayFrame(null);
            setTimeout(() => {
                if (ws) {
                    ws.close();
                }
            }, 3000);
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

    useEffect(() => {
        if (Object.keys(annotationData).length > 0) {
            const annotationVideoNames = Object.keys(annotationData).sort();
            const uploadedVideoNames = uploadedFiles.filter(f => f && f.name).map(f => f.name).sort();
            const isSameLength = annotationVideoNames.length === uploadedVideoNames.length;
            const isSameNames = annotationVideoNames.every((name, idx) => name === uploadedVideoNames[idx]);
            setIsAnnotationVideoMismatch(!isSameLength || !isSameNames);
        }
    }, [uploadedFiles, annotationData]);

    // NAS 폴더 복사 진행 상태 폴링
    useEffect(() => {
        let interval: NodeJS.Timeout;
        if (showNasProgress) {
            interval = setInterval(async () => {
                try {
                    const response = await fetch('/nas_copy_progress');
                    if (response.ok) {
                        const progress = await response.json();
                        setNasCopyProgress(progress);
                        // 복사가 완료되면 파일 목록 새로고침 및 진행률/입력창 UI 숨김
                        if (!progress.is_copying && progress.copied_files >= progress.total_files && progress.total_files > 0) {
                            const filesResponse = await fetch('/uploads');
                            if (filesResponse.ok) {
                                const data = await filesResponse.json();
                                if (Array.isArray(data.files)) {
                                    setUploadedFiles(data.files);
                                }
                            }
                            setShowNasProgress(false);
                            setNasInputOpen(false);
                        }
                    }
                } catch (error) {
                    console.error('NAS 복사 진행 상태 조회 실패:', error);
                }
            }, 1000);
        }
        return () => {
            if (interval) {
                clearInterval(interval);
            }
        };
    }, [showNasProgress]);

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
            const modelRes = await fetch('/model', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_id: id }),
            });
            if (!modelRes.ok) throw new Error('모델 로딩 실패');
            const data = await modelRes.json();
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
        let files = e.target.files;
        if (!files || files.length === 0) return;

        // 업로드 시작 전 이미 업로드된 파일명 Set으로 저장
        let alreadyUploadedNames = new Set<string>();
        try {
            const res = await fetch('/uploads');
            if (res.ok) {
                const data = await res.json();
                if (Array.isArray(data.files)) {
                    data.files.forEach((f: UploadedFile) => {
                        if (f && f.name) alreadyUploadedNames.add(f.name);
                    });
                }
            }
        } catch {}

        // Filter only video files (mp4, avi, mkv)
        const allowedExtensions = [".mp4", ".avi", ".mkv"];
        // 중복 파일명은 업로드 시도 자체를 하지 않음
        const filteredFiles = Array.from(files).filter(file => file && file.name && allowedExtensions.some(ext => file.name.toLowerCase().endsWith(ext)) && !alreadyUploadedNames.has(file.name));
        if (filteredFiles.length === 0) {
            alert("업로드할 새로운 mp4, avi, mkv 파일이 없습니다.");
            e.target.value = '';
            return;
        }

        // 전체 파일 수 = 이미 업로드된 + 실제 새로 업로드할 파일
        setTotalFilesToUpload(alreadyUploadedNames.size + filteredFiles.length);
        let uploadedCount = alreadyUploadedNames.size;
        setUploadedFileCount(uploadedCount);
        setCurrentUploadSessionFiles([]);

        for (let i = 0; i < filteredFiles.length; i++) {
            const file = filteredFiles[i];
            if (!file || !file.name) continue;
            const formData = new FormData();
            formData.append('files', file);
            formData.append('paths', file.webkitRelativePath || file.name);
            try {
                const uploadRes = await fetch('/upload', { method: 'POST', body: formData });
                if (!uploadRes.ok) throw new Error(`파일 업로드 실패: ${file && file.name}`);
                const data = await uploadRes.json();
                setUploadedFiles(prev => {
                    const newFiles = [...prev, ...(Array.isArray(data.files) ? data.files.filter((f: UploadedFile) => f && f.name) : [])];
                    return newFiles.filter((v, idx, a) => v && v.name && a.findIndex(t => t && t.name === v.name) === idx);
                });
                // 실제 업로드된 파일만 카운트
                if (Array.isArray(data.files) && data.files.length > 0) {
                    uploadedCount += data.files.length;
                }
                setUploadedFileCount(uploadedCount);
                if (Array.isArray(data.files) && data.files[0] && data.files[0].name) {
                    setCurrentUploadSessionFiles(prev => [...prev, data.files[0].name]);
                }
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
        const res = await fetch(`/upload/${encodeURIComponent(fileName)}`, { method: 'DELETE' });
        if (res.ok) {
        setUploadedFiles(prev => prev.filter((f: UploadedFile) => f && f.name && f.name !== fileName));
        if (selectedUploadedFileName === fileName) {
            setSelectedUploadedFileName('');
            }
        }
    };

    const handleRemoveAllFiles = async () => {
        const res = await fetch('/uploads', { method: 'DELETE' });
        if (res.ok) {
        setUploadedFiles([]);
        setSelectedUploadedFileName('');
        }
    };

    const handleFileSelectChange = (event: SelectChangeEvent<string>) => {
        setSelectedUploadedFileName(event.target.value as string);
    };

    const handleStartInference = () => {
        setCumulativeAccuracyHistory([]);
        setMetricsHistory([]);
        const body = {
            interval: frameInterval,
            infer_period: inferPeriod,
            batch: batchFrames,
            inference_mode: inferenceMode,
            annotation_data: annotationData
        };
        fetch('/infer', {
            method: 'POST', 
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
    };

    const handlePauseInference = () => {
        fetch('/pause_infer', { method: 'POST' });
        setIsPaused(true);
    };

    const handleResumeInference = () => {
        fetch('/resume_infer', { method: 'POST' });
        setIsPaused(false);
    };

    const handleStopInference = () => {
        fetch('/stop_infer', { method: 'POST' });
        setIsPaused(false);
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
                    
                    // === 업로드된 비디오와 비교 ===
                    const annotationVideoNames = Object.keys(parsedData).sort();
                    const uploadedVideoNames = uploadedFiles.filter(f => f && f.name).map(f => f.name).sort();
                    const isSameLength = annotationVideoNames.length === uploadedVideoNames.length;
                    const isSameNames = annotationVideoNames.every((name, idx) => name === uploadedVideoNames[idx]);
                    if (!isSameLength || !isSameNames) {
                        setIsAnnotationVideoMismatch(true);
                        setAnnotationAlertOpen(true);
                        return;
                    }
                    setAnnotationData(parsedData);
                    setIsAnnotationVideoMismatch(false);
                    setAnnotationAlertOpen(false);
                    setAnnotationSuccessOpen(true);
                } catch (error: any) {
                    console.error("어노테이션 파일 파싱 오류:", error);
                    alert(`어노테이션 파일 파싱에 실패했습니다: ${error.message}`);
                    setAnnotationData({});
                } finally {
                    e.target.value = '';
                }
            };
            reader.readAsText(file);
        } else {
            e.target.value = '';
        }
    };

    const handleResultFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            setSelectedResultFile(file);
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
    
    const isAnalysisComplete = !!inferenceState && Array.isArray(inferenceState.events) && inferenceState.events.some(ev => ev.type === 'complete');
    
    const handleOpenAnalysisVideoSelect = async () => {
        try {
            const resultsRes = await fetch('/results/videos');
            if (!resultsRes.ok) throw new Error('결과 비디오 로딩 실패');
            const data = await resultsRes.json();
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

    // 모델 해제 핸들러
    const handleUnloadModel = async () => {
        try {
            const res = await fetch('/unload_model', { method: 'POST' });
            if (!res.ok) throw new Error('모델 해제 실패');
            setModelId('');
            setModelStatus('none');
        } catch (e) {
            alert('모델 해제 실패');
        }
    };

    // 2. NAS 폴더 버튼 클릭 시 입력창 토글
    const handleNasFolderSelect = () => {
        setNasInputOpen((prev) => !prev);
    };

    // 3. NAS 폴더 업로드 버튼 클릭 시
    const handleNasFolderUpload = async () => {
        const trimmedPath = nasInputValue.trim();
        if (!trimmedPath) {
            alert('경로를 입력하세요.');
            return;
        }
        if (!(trimmedPath.startsWith(nasTargetPath) || trimmedPath.startsWith('./') || trimmedPath.startsWith('../') || !trimmedPath.startsWith('/'))) {
            alert(`경로 형식이 올바르지 않습니다. 절대 경로(${nasTargetPath}) 또는 상대 경로(./, ../)를 사용하세요.`);
            return;
        }
        setShowNasProgress(true);
        setNasInputOpen(true);
        await processNasFolder(trimmedPath);
    };

    const processNasFolder = async (folderPath: string) => {
        try {
            // 백엔드로 폴더 경로 전송
            const response = await fetch('/process_nas_folder', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    nas_folder: folderPath
                }),
            });
            
            if (response.ok) {
                const result = await response.json();
                setNasCopyProgress({
                    is_copying: true,
                    total_files: result.total_files,
                    copied_files: result.copied_files,
                    current_file: null,
                    errors: result.errors
                });
                setNasCopyProgressOpen(true);
                
                if (result.errors && result.errors.length > 0) {
                    alert(`복사 중 오류가 발생했습니다:\n${result.errors.join('\n')}`);
                }
            } else {
                const error = await response.json();
                alert(`NAS 폴더 처리 실패: ${error.detail || '알 수 없는 오류'}`);
            }
        } catch (error) {
            console.error('NAS 폴더 처리 실패:', error);
            alert('NAS 폴더 처리 중 오류가 발생했습니다.');
        }
    };

    const handleCancelNasCopy = async () => {
        try {
            const response = await fetch('/cancel_nas_copy', {
                method: 'POST',
            });
            if (response.ok) {
                setNasCopyProgress(prev => ({ ...prev, is_copying: false }));
                setNasCopyProgressOpen(false);
                // 업로드된 파일도 모두 삭제
                await fetch('/uploads', { method: 'DELETE' });
                setUploadedFiles([]);
                setShowNasProgress(false);
                setNasInputOpen(false);
            }
        } catch (error) {
            console.error('NAS 복사 취소 실패:', error);
        }
    };

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
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                                <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>Video Upload</Typography>
                                <Box></Box>
                            </Box>
                            {totalFilesToUpload > 0 ? (
                                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                                    업로드 중: {uploadedFileCount} / {totalFilesToUpload} ({((uploadedFileCount / totalFilesToUpload) * 100).toFixed(0)}%)
                                </Typography>
                            ) : (
                                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                                    {uploadedFiles.length > 0 
                                        ? `현재 업로드된 비디오: ${uploadedFiles.length}개`
                                        : '업로드된 비디오가 없습니다.'
                                    }
                                </Typography>
                            )}
                            <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
                                <Button variant="outlined" component="label" startIcon={<FolderOpenIcon />} size="small" sx={{ flex: 1 }}>파일 열기<input type="file" hidden multiple onChange={handleFileUpload} /></Button>
                                <Button variant="outlined" component="label" startIcon={<FolderOpenIcon />} size="small" sx={{ flex: 1 }}>폴더 열기<input type="file" hidden multiple webkitdirectory="" onChange={handleFileUpload} /></Button>
                                <Button variant="outlined" startIcon={<FolderOpenIcon />} size="small" sx={{ flex: 1 }} onClick={handleNasFolderSelect}>NAS 폴더</Button>
                                <Button variant="text" color="error" size="small" onClick={handleRemoveAllFiles}>전체 삭제</Button>
                            </Box>
                            {nasInputOpen && (
                                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, mb: 2, alignItems: 'stretch' }}>
                                    <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                                        <TextField
                                            label="NAS 폴더 경로"
                                            value={nasInputValue}
                                            onChange={e => setNasInputValue(e.target.value)}
                                            size="small"
                                            fullWidth
                                        />
                                        <Button variant="contained" color="primary" onClick={handleNasFolderUpload}>Upload</Button>
                                    </Box>
                                    {showNasProgress && (
                                        <Box sx={{ width: '100%', mt: 1 }}>
                                            <LinearProgress
                                                variant="determinate"
                                                value={nasCopyProgress.total_files > 0 ? (nasCopyProgress.copied_files / nasCopyProgress.total_files) * 100 : 0}
                                                sx={{ height: 10, borderRadius: 5 }}
                                            />
                                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 0.5 }}>
                                                <Typography variant="body2">
                                                    복사 진행률: {nasCopyProgress.copied_files} / {nasCopyProgress.total_files}
                                                </Typography>
                                                <Button variant="outlined" color="error" size="small" onClick={handleCancelNasCopy} sx={{ ml: 2 }}>
                                                    복사 취소
                                                </Button>
                                            </Box>
                                        </Box>
                                    )}
                                </Box>
                            )}
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
                                    {uploadedFiles.filter((f: UploadedFile) => f && f.name).map((f: UploadedFile) => (
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
                                                <IconButton
                                                    edge="end"
                                                    aria-label="delete"
                                                    onClick={(e) => {
                                                        e.stopPropagation(); // Prevent MenuItem from closing
                                                        handleRemoveFile(f.name);
                                                    }}
                                                    sx={{ ml: 'auto' }} // Push to the right
                                                >
                                                    <DeleteIcon sx={{ fontSize: 20 }} />
                                                </IconButton>
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
                                        fullWidth sx={{ fontSize: '0.7rem', input: { textAlign: 'center' } }}
                                    />
                                </Grid>
                                <Grid item xs={4}>
                                    <TextField
                                        label="추론 주기(Frames)"
                                        type="number"
                                        value={inferPeriod}
                                        onChange={(e) => setInferPeriod(Number(e.target.value))}
                                        size="small"
                                        fullWidth sx={{ fontSize: '0.7rem', input: { textAlign: 'center' } }}
                                    />
                                </Grid>
                                <Grid item xs={4}>
                                    <TextField
                                        label="추출 프레임(Batch)"
                                        type="number"
                                        value={batchFrames}
                                        onChange={(e) => setBatchFrames(Number(e.target.value))}
                                        size="small"
                                        fullWidth sx={{ fontSize: '0.7rem', input: { textAlign: 'center' } }}
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
                                    <Button 
                                        variant={annotationData && Object.keys(annotationData).length > 0 ? "contained" : "outlined"}
                                        color={annotationData && Object.keys(annotationData).length > 0 ? "primary" : "inherit"}
                                        component="label" size="small" fullWidth sx={{ fontSize: '0.8rem' }}>
                                        Annotation 
                                        <input type="file" hidden accept=".json,.txt" onChange={handleAnnotationUpload} />
                                    </Button>
                                </Grid>
                            </Grid>
                            {isInferring && !isPaused ? (
                                <Button
                                    variant="contained"
                                    color="warning"
                                    onClick={handlePauseInference}
                                    fullWidth
                                    sx={{ mt: 0, mb: 1 }}
                                >
                                    PAUSE
                                </Button>
                            ) : isInferring && isPaused ? (
                                <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
                                    <Button
                                        variant="contained"
                                        color="error"
                                        onClick={handleStopInference}
                                        fullWidth
                                    >
                                        STOP
                                    </Button>
                                    <Button
                                        variant="contained"
                                        color="primary"
                                        onClick={handleResumeInference}
                                        fullWidth
                                    >
                                        RESUME
                                    </Button>
                                </Box>
                            ) : (
                                <Button
                                    variant="contained"
                                    color="primary"
                                    onClick={handleStartInference}
                                    disabled={
                                      !modelId ||
                                      uploadedFiles.length === 0 ||
                                      isAnnotationVideoMismatch ||
                                      !annotationData || Object.keys(annotationData).length === 0 ||
                                      (inferenceMode !== 'AR' && inferenceMode !== 'AL')
                                    }
                                    fullWidth
                                    sx={{ mt: 0, mb: 1 }}
                                >
                                    RUN
                                </Button>
                            )}
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
                                        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', flexShrink: 0, mb: 1, position: 'relative' }}>
                                            <Typography variant="subtitle1" color="white" sx={{ textAlign: 'center', width: '100%' }}>
                                                추론 결과 분석
                                            </Typography>
                                            <Button
                                                variant="contained"
                                                size="small"
                                                component="label"
                                                sx={{ position: 'absolute', right: 0 }}
                                            >
                                                Open videos
                                                <input
                                                    type="file"
                                                    accept=".mp4"
                                                    style={{ display: 'none' }}
                                                    onChange={handleResultFileChange}
                                                />
                                            </Button>
                                        </Box>
                                        <Box sx={{ 
                                            flex: 1, 
                                            width: '100%', 
                                            display: 'flex', 
                                            justifyContent: 'center', 
                                            alignItems: 'center', 
                                            minHeight: 0
                                        }}>
                                            {selectedResultFile ? (
                                                <VideoPlayer videoUrl={URL.createObjectURL(selectedResultFile)} />
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
                                                    <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                                                        ※ NAS(aivanas) save_results 폴더 내의 mp4 파일만 선택해 주세요.
                                                    </Typography>
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
                                    <Grid item xs={12} md={6} sx={{ display: 'flex', flexDirection: 'column', height: '100%', alignItems: 'center' }}>
                                        <Box sx={{ mt: 7 }}>
                                            <Typography variant="subtitle2" gutterBottom sx={{ mb: 4, fontSize: '1.2em', width: '90%', textAlign: 'center', whiteSpace: 'nowrap' }}>Current Inferencing Video</Typography>
                                            <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center' }}>
                                                {inferenceState.current_video ? inferenceState.current_video : '추론 중인 비디오 없음'}
                                            </Typography>
                                        </Box>
                                        <ConfusionMatrixDisplay metrics={metricsHistory.length > 0 ? metricsHistory[metricsHistory.length - 1] : { tp: 0, tn: 0, fp: 0, fn: 0, precision: 0, recall: 0, f1_score: 0 }} />
                                    </Grid>

                                    {/* 오른쪽 절반 - 성능 지표 및 누적 정확도 그래프 */}
                                    <Grid item xs={12} md={6} sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                                        <MetricsBarChart metrics={metricsHistory.length > 0 ? metricsHistory[metricsHistory.length - 1] : { tp: 0, tn: 0, fp: 0, fn: 0, precision: 0, recall: 0, f1_score: 0 }} />
                                        <CumulativeAccuracyGraph cumulativeAccuracyHistory={Array.isArray(cumulativeAccuracyHistory) ? cumulativeAccuracyHistory : []} />
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
                            <ConfusionMatrixGraph metrics={inferenceState && inferenceState.metrics ? inferenceState.metrics : { tp: 0, tn: 0, fp: 0, fn: 0, precision: 0, recall: 0, f1_score: 0 }} />
                        </Box>
                    </Grid>
                </Grid>
                <Dialog open={annotationAlertOpen} onClose={() => setAnnotationAlertOpen(false)}>
                    <DialogTitle>어노테이션/비디오 불일치</DialogTitle>
                    <DialogContent>
                        <Typography>어노테이션 파일의 비디오 목록과 업로드된 비디오 목록이 일치하지 않습니다!</Typography>
                    </DialogContent>
                    <DialogActions>
                        <Button onClick={() => setAnnotationAlertOpen(false)}>확인</Button>
                    </DialogActions>
                </Dialog>
                <Dialog open={annotationSuccessOpen} onClose={() => setAnnotationSuccessOpen(false)}>
                    <DialogTitle>어노테이션 성공</DialogTitle>
                    <DialogContent>
                        <Typography>어노테이션 파일이 성공적으로 로드되었습니다.</Typography>
                    </DialogContent>
                    <DialogActions>
                        <Button onClick={() => setAnnotationSuccessOpen(false)}>확인</Button>
                    </DialogActions>
                </Dialog>
            </Container>
        </ThemeProvider>
    );
}

export default App;
