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
  const detectionEvents = events.filter(e => e.type === 'detection' && e.data).reverse();

  return (
    <Paper sx={{ width: '100%', p: 2, display: 'flex', flexDirection: 'column', minHeight: 0, height: '100%' }}>
      <Typography variant="h6" gutterBottom sx={{ flexShrink: 0 }}>
        Inference Results
      </Typography>
      <TableContainer sx={{ height: 'calc(100% - 62px)', overflow: 'auto' }}>
        <Table stickyHeader size="small">
          <TableHead>
            <TableRow>
              <TableCell sx={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>비디오 클립</TableCell>
              <TableCell align="right" sx={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>시작(초)</TableCell>
              <TableCell align="right" sx={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>종료(초)</TableCell>
              <TableCell sx={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>이벤트</TableCell>
              <TableCell align="right" sx={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>시작 프레임</TableCell>
              <TableCell align="right" sx={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>종료 프레임</TableCell>
              <TableCell align="right" sx={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>추론 시간</TableCell>
              <TableCell align="right" sx={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>추론 FPS</TableCell>
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
                  <TableCell align="right">{event.data.start_frame}</TableCell>
                  <TableCell align="right">{event.data.end_frame}</TableCell>
                  <TableCell align="right">{event.data.inference_time_ms}</TableCell>
                  <TableCell align="right">{event.data.inference_fps}</TableCell>
                </TableRow>
              );
            })}
            {detectionEvents.length === 0 && (
              <TableRow>
                <TableCell colSpan={8} align="left">추론 이벤트를 기다리는 중...</TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>
    </Paper>
  );
};

export default InferenceResultTable;
