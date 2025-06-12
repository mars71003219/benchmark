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
      <Box sx={{ p: 2, textAlign: 'center', color: 'text.secondary' }}>
        추론 메트릭을 기다리는 중...
      </Box>
    );
  }

  return (
    <TableContainer component={Paper} sx={{ mt: 2, p: 2 }}>
      <Typography variant="h6" gutterBottom>실시간 추론 메트릭</Typography>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>지표</TableCell>
            <TableCell align="right">값</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          <TableRow>
            <TableCell>True Positive (TP)</TableCell>
            <TableCell align="right">{metrics.tp}</TableCell>
          </TableRow>
          <TableRow>
            <TableCell>True Negative (TN)</TableCell>
            <TableCell align="right">{metrics.tn}</TableCell>
          </TableRow>
          <TableRow>
            <TableCell>False Positive (FP)</TableCell>
            <TableCell align="right">{metrics.fp}</TableCell>
          </TableRow>
          <TableRow>
            <TableCell>False Negative (FN)</TableCell>
            <TableCell align="right">{metrics.fn}</TableCell>
          </TableRow>
          <TableRow>
            <TableCell>정확도 (Accuracy)</TableCell>
            <TableCell align="right">{((metrics.tp + metrics.tn) / (metrics.tp + metrics.tn + metrics.fp + metrics.fn) || 0).toFixed(2)}</TableCell>
          </TableRow>
          <TableRow>
            <TableCell>정밀도 (Precision)</TableCell>
            <TableCell align="right">{metrics.precision.toFixed(2)}</TableCell>
          </TableRow>
          <TableRow>
            <TableCell>재현율 (Recall)</TableCell>
            <TableCell align="right">{metrics.recall.toFixed(2)}</TableCell>
          </TableRow>
          <TableRow>
            <TableCell>F1-Score</TableCell>
            <TableCell align="right">{metrics.f1_score.toFixed(2)}</TableCell>
          </TableRow>
        </TableBody>
      </Table>
    </TableContainer>
  );
};

export default ConfusionMatrixDisplay; 