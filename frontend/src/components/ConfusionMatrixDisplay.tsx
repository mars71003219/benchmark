import React from 'react';
import {
  Box, Typography, Paper, Table, TableBody, TableCell,
  TableContainer, TableHead, TableRow
} from '@mui/material';

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

const ConfusionMatrixDisplay: React.FC<ConfusionMatrixDisplayProps> = ({ metrics }) => {
  if (!metrics) {
    return (
      <Box sx={{ p: 2, textAlign: 'left', color: 'text.secondary' }}>
        추론 메트릭을 기다리는 중...
      </Box>
    );
  }

  return (
    <TableContainer component={Paper} sx={{ mt: 10, p: 2 }}>
      <Typography variant="h6" gutterBottom sx={{ textAlign: 'center' }}>Model Evaluation Metrics</Typography>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell sx={{ height: '40px' }}>지표</TableCell>
            <TableCell align="right" sx={{ height: '40px' }}>값</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          <TableRow>
            <TableCell sx={{ height: '40px' }}>True Positive (TP)</TableCell>
            <TableCell align="right" sx={{ height: '40px' }}>{metrics.tp}</TableCell>
          </TableRow>
          <TableRow>
            <TableCell sx={{ height: '40px' }}>True Negative (TN)</TableCell>
            <TableCell align="right" sx={{ height: '40px' }}>{metrics.tn}</TableCell>
          </TableRow>
          <TableRow>
            <TableCell sx={{ height: '40px' }}>False Positive (FP)</TableCell>
            <TableCell align="right" sx={{ height: '40px' }}>{metrics.fp}</TableCell>
          </TableRow>
          <TableRow>
            <TableCell sx={{ height: '40px' }}>False Negative (FN)</TableCell>
            <TableCell align="right" sx={{ height: '40px' }}>{metrics.fn}</TableCell>
          </TableRow>
          <TableRow>
            <TableCell sx={{ height: '40px' }}>정확도 (Accuracy)</TableCell>
            <TableCell align="right" sx={{ height: '40px' }}>{((metrics.tp + metrics.tn) / (metrics.tp + metrics.tn + metrics.fp + metrics.fn) || 0).toFixed(2)}</TableCell>
          </TableRow>
          <TableRow>
            <TableCell sx={{ height: '40px' }}>정밀도 (Precision)</TableCell>
            <TableCell align="right" sx={{ height: '40px' }}>{metrics.precision.toFixed(2)}</TableCell>
          </TableRow>
          <TableRow>
            <TableCell sx={{ height: '40px' }}>재현율 (Recall)</TableCell>
            <TableCell align="right" sx={{ height: '40px' }}>{metrics.recall.toFixed(2)}</TableCell>
          </TableRow>
          <TableRow>
            <TableCell sx={{ height: '40px' }}>F1-Score</TableCell>
            <TableCell align="right" sx={{ height: '40px' }}>{metrics.f1_score.toFixed(2)}</TableCell>
          </TableRow>
        </TableBody>
      </Table>
    </TableContainer>
  );
};

export default ConfusionMatrixDisplay; 