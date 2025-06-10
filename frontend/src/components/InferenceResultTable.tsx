// frontend/src/components/InferenceResultTable.tsx

import React from 'react';
import {
  Paper, Table, TableBody, TableCell,
  TableContainer, TableHead, TableRow, Typography,
} from '@mui/material';

interface InferenceEvent {
  type: string;
  [key: string]: any;
}

interface InferenceResultTableProps {
  events: InferenceEvent[];
  classLabels: string[];
}

const InferenceResultTable: React.FC<InferenceResultTableProps> = ({ events, classLabels }) => {
  const detectionEvents = events.filter(e => e.type === 'detection' && e.data);

  return (
    <Paper sx={{ width: '100%', flexGrow: 1, p: 2, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
      <Typography variant="h6" gutterBottom sx={{ flexShrink: 0 }}>
        실시간 추론 이벤트
      </Typography>
      <TableContainer sx={{ flexGrow: 1, overflow: 'auto', maxHeight: '100%' }}>
        <Table stickyHeader size="small">
          <TableHead>
            <TableRow>
              <TableCell>비디오 클립</TableCell>
              <TableCell align="right">시작(초)</TableCell>
              <TableCell align="right">종료(초)</TableCell>
              <TableCell>이벤트</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {detectionEvents.map((event, index) => {
              const predictionLabel = classLabels[event.data.prediction_label_id] || event.data.prediction_label;
              return (
                <TableRow key={`${event.data.video_name}-${index}`} hover>
                  <TableCell component="th" scope="row">{event.data.video_name}</TableCell>
                  <TableCell align="right">{event.data.start_time.toFixed(2)}</TableCell>
                  <TableCell align="right">{event.data.end_time.toFixed(2)}</TableCell>
                  <TableCell>{predictionLabel}</TableCell>
                </TableRow>
              );
            })}
            {detectionEvents.length === 0 && (
              <TableRow>
                <TableCell colSpan={4} align="center">추론 이벤트를 기다리는 중...</TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>
    </Paper>
  );
};

export default InferenceResultTable;
