import React from 'react';
import { Paper, Typography } from '@mui/material';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement, Tooltip, Legend);

interface Event {
  timestamp: string;
  type: string;
  data?: any; // Add data field if it exists in your event structure
}

interface Props {
  events: Event[];
  classLabels: string[];
  videoDuration: number;
  cumulativeAccuracyHistory: { processed_clips: number; accuracy: number; }[]; // New prop for cumulative accuracy
}

const VideoInferenceChart: React.FC<Props> = ({ events, classLabels, videoDuration, cumulativeAccuracyHistory }) => {
  // x축: 0 ~ videoDuration (초)
  const xTicks = Array.from({length: Math.ceil(videoDuration)+1}, (_, i) => i);

  // y축: classLabels
  // 각 라벨별로 시간별 카운트(혹은 1/0) 데이터 생성
  const labelData: {[label: string]: number[]} = {};
  classLabels.forEach(label => {
    labelData[label] = xTicks.map(() => 0);
  });
  events.forEach(ev => {
    // ev에 라벨 정보가 있다고 가정 (없으면 무시)
    // ev.type === 'detection' && ev.data.prediction_label, ev.data.start_time
    // start_time(초) 기준으로 해당 라벨에 +1
    // (실제 데이터 구조에 맞게 조정 필요)
    // 예시: ev = { type: 'detection', data: { prediction_label: 'car', start_time: 12.3 } }
    if (ev.type === 'detection' && ev.data?.prediction_label && ev.data?.start_time !== undefined) {
      const label = ev.data.prediction_label;
      const t = Math.floor(ev.data.start_time);
      if (labelData[label] && t >= 0 && t < xTicks.length) {
        labelData[label][t] += 1;
      }
    }
  });

  // 누적 정확도 데이터셋 추가
  const cumulativeAccuracyDataset = {
    label: '누적 정확도',
    data: cumulativeAccuracyHistory.map(item => item.accuracy),
    borderColor: 'rgb(75, 192, 192)',
    backgroundColor: 'rgba(75, 192, 192, 0.2)',
    fill: false,
    tension: 0.3,
    pointRadius: 3,
    yAxisID: 'y1',
  };

  const data = {
    labels: xTicks,
    datasets: [
      ...classLabels.map((label, idx) => ({
        label,
        data: labelData[label],
        borderColor: `hsl(${(idx*47)%360},70%,50%)`,
        backgroundColor: `hsla(${(idx*47)%360},70%,50%,0.2)`,
        fill: false,
        tension: 0.3,
        pointRadius: 2,
        yAxisID: 'y0',
      })),
      cumulativeAccuracyDataset,
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: { display: true },
      tooltip: { enabled: true },
    },
    scales: {
      y0: {
        type: 'linear' as const,
        position: 'left' as const,
        title: { display: true, text: '클래스별 이벤트 수' },
        min: 0,
        ticks: { stepSize: 1 },
      },
      y1: {
        type: 'linear' as const,
        position: 'right' as const,
        title: { display: true, text: '누적 정확도' },
        min: 0,
        max: 1,
        ticks: { stepSize: 0.1 },
        grid: { drawOnChartArea: false }, // Only draw grid lines for y0
      },
      x: {
        title: { display: true, text: '비디오 시간(초)' },
        min: 0,
        max: videoDuration,
        ticks: { stepSize: Math.max(1, Math.floor(videoDuration/10)) },
      },
    },
  } as const;

  return (
    <Paper sx={{ p: 3, mb: 3, height: 300 }}>
      <Typography variant="h6" gutterBottom>
        비디오 추론 이벤트 그래프
      </Typography>
      <Line data={data} options={options} />
    </Paper>
  );
};

export default VideoInferenceChart; 