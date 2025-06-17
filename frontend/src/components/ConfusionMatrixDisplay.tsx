import React from 'react';
import {
  Box, Typography, Paper, Table, TableBody, TableCell,
  TableContainer, TableHead, TableRow
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingFlatIcon from '@mui/icons-material/TrendingFlat';

interface Metrics {
  tp: number;
  tn: number;
  fp: number;
  fn: number;
  precision: number;
  recall: number;
  f1_score: number;
}

interface ConfusionMatrixDisplayProps {
  metrics?: Metrics;
}

const metricRows = (metrics: Metrics) => [
  { label: 'True Positive (TP)', value: metrics.tp },
  { label: 'True Negative (TN)', value: metrics.tn },
  { label: 'False Positive (FP)', value: metrics.fp },
  { label: 'False Negative (FN)', value: metrics.fn },
  { label: '정확도 (Accuracy)', value: ((metrics.tp + metrics.tn) / (metrics.tp + metrics.tn + metrics.fp + metrics.fn) || 0).toFixed(2) },
  { label: '정밀도 (Precision)', value: metrics.precision.toFixed(2) },
  { label: '재현율 (Recall)', value: metrics.recall.toFixed(2) },
  { label: 'F1-Score', value: metrics.f1_score.toFixed(2) },
];

const ConfusionMatrixDisplay: React.FC<ConfusionMatrixDisplayProps> = ({ metrics }) => {
  if (!metrics) {
    return (
      <Box sx={{ p: 2, textAlign: 'left', color: 'text.secondary' }}>
        추론 메트릭을 기다리는 중...
      </Box>
    );
  }

  return (
    <TableContainer component={Paper} elevation={4} sx={{ mt: 4, borderRadius: 3, boxShadow: 3, p: 2 }}>
      <Typography variant="h6" gutterBottom sx={{ textAlign: 'center', fontWeight: 'bold', mb: 2 }}>
        Model Evaluation Metrics
      </Typography>
      <Table size="small">
        <TableHead>
          <TableRow sx={{ background: '#f5f7fa' }}>
            <TableCell sx={{ height: '40px', fontWeight: 'bold', fontSize: 16 }}>지표</TableCell>
            <TableCell align="center" sx={{ height: '40px', fontWeight: 'bold', fontSize: 16 }}>값</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {metricRows(metrics).map((row, idx) => (
            <TableRow hover key={row.label} sx={{ transition: 'background 0.2s', '&:hover': { background: '#e3f2fd' } }}>
              <TableCell sx={{ height: '40px', fontWeight: idx < 4 ? 'bold' : 'normal' }}>{row.label}</TableCell>
              <TableCell align="center" sx={{ height: '40px', fontWeight: 'bold', color: 'black' }}>{row.value}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

export default ConfusionMatrixDisplay; 