import React, { useCallback, useRef, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Box, Typography, Paper, Button, Stack } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

interface VideoUploaderProps {
  onUpload: (files: File[]) => void;
}

const isLinux = navigator.platform.toLowerCase().includes('linux');

const VideoUploader: React.FC<VideoUploaderProps> = ({ onUpload }) => {
  const [showFileInput, setShowFileInput] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    onUpload(acceptedFiles);
  }, [onUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple: true
  });

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      onUpload(Array.from(e.target.files));
      setShowFileInput(false); // input unmount
    }
  };

  // 폴더 업로드: webkitdirectory input만 사용
  const handleFolderUpload = () => {
    document.getElementById('folder-fallback-input')?.click();
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        비디오 파일/폴더 업로드
      </Typography>
      <Paper
        {...getRootProps()}
        sx={{
          p: 3,
          textAlign: 'center',
          cursor: 'pointer',
          bgcolor: isDragActive ? 'action.hover' : 'background.paper',
          border: '2px dashed',
          borderColor: isDragActive ? 'primary.main' : 'divider',
          '&:hover': {
            bgcolor: 'action.hover'
          }
        }}
      >
        <input {...getInputProps()} />
        <CloudUploadIcon sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
        <Typography>
          {isDragActive
            ? '여기에 파일 또는 폴더를 놓으세요'
            : '파일을 드래그하거나 아래 버튼을 이용해 업로드하세요'}
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          지원 형식: MP4, AVI, MOV<br />
          {isLinux
            ? <b>리눅스 환경에서는 여러 파일을 한 번에 선택해 업로드하세요.</b>
            : <b>폴더 업로드는 Chromium 기반 브라우저에서만 하위 폴더까지 지원됩니다.</b>
          }
        </Typography>
        <Stack direction="row" spacing={2} justifyContent="center" sx={{ mt: 2 }}>
          <Button
            variant="contained"
            onClick={() => setShowFileInput(true)}
          >
            파일 업로드
          </Button>
          {!isLinux && (
            <Button
              variant="outlined"
              onClick={handleFolderUpload}
            >
              폴더 업로드
            </Button>
          )}
        </Stack>
        {showFileInput && (
          <input
            type="file"
            ref={fileInputRef}
            style={{ display: 'none' }}
            multiple
            onChange={handleFileChange}
            accept=".mp4,.avi,.mov,video/*"
            onClick={e => (e.currentTarget.value = '')}
            autoFocus
          />
        )}
        {/* fallback: webkitdirectory input */}
        <input
          id="folder-fallback-input"
          type="file"
          style={{ display: 'none' }}
          // @ts-ignore
          webkitdirectory="true"
          // @ts-ignore
          directory="true"
          onChange={e => {
            if (e.target.files) {
              const files = Array.from(e.target.files).filter(file =>
                file.type.startsWith('video/') ||
                file.name.endsWith('.mp4') ||
                file.name.endsWith('.avi') ||
                file.name.endsWith('.mov')
              );
              onUpload(files);
            }
          }}
        />
      </Paper>
    </Box>
  );
};

export default VideoUploader; 