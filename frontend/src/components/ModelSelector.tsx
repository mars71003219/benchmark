import React, { useState } from 'react';
import {
  Box,
  TextField,
  Button,
  Typography,
  Alert
} from '@mui/material';
import PsychologyIcon from '@mui/icons-material/Psychology';

interface ModelSelectorProps {
  onSubmit: (url: string) => void;
}

const ModelSelector: React.FC<ModelSelectorProps> = ({ onSubmit }) => {
  const [modelId, setModelId] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!modelId) {
      setError('모델 ID를 입력해주세요');
      return;
    }

    // 허깅페이스 모델 ID 형식: username/model_name
    if (!/^([\w-]+)\/[\w\-.]+$/.test(modelId)) {
      setError('유효한 허깅페이스 모델 ID를 입력해주세요. 예: Jeyseb/videomae-base-finetuned-rwf2000-subset___v4');
      return;
    }

    setError('');
    onSubmit(modelId);
  };

  return (
    <Box component="form" onSubmit={handleSubmit}>
      <Typography variant="subtitle1" gutterBottom sx={{ mb: 2, textAlign: 'left', fontWeight: 'bold' }}>
       HuggingFace Model Selector
      </Typography>

      <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
        <TextField
          fullWidth
          label="모델 ID"
          variant="outlined"
          value={modelId}
          onChange={(e) => setModelId(e.target.value)}
          placeholder="예: Jeyseb/videomae-base-finetuned-rwf2000-subset___v4"
          error={!!error}
          helperText={error}
          size="small"
          InputProps={{
            startAdornment: <PsychologyIcon sx={{ mr: 1, color: 'primary.main' }} />,
            sx: { py: 0.5 }
          }}
        />
        
        <Button
          type="submit"
          variant="contained"
          size="small"
          sx={{ minWidth: 100 }}
        >
          모델 로드
        </Button>
      </Box>
    </Box>
  );
};

export default ModelSelector; 