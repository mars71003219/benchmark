import React from 'react';
import { Box, Typography } from '@mui/material';

interface MetricsBarChartProps {
    metrics: {
        tp: number;
        tn: number;
        fp: number;
        fn: number;
        precision: number;
        recall: number;
        f1_score: number;
    } | undefined;
}

const MetricsBarChart: React.FC<MetricsBarChartProps> = ({ metrics }) => {
    if (!metrics) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', width: '100%' }}>
                <Typography variant="body2" color="text.secondary">메트릭스 데이터 없음</Typography>
            </Box>
        );
    }

    // Calculate accuracy
    const total = metrics.tp + metrics.tn + metrics.fp + metrics.fn;
    const accuracy = total > 0 ? (metrics.tp + metrics.tn) / total : 0;

    const data = [
        { label: 'Accuracy', value: accuracy, color: '#1f77b4' }, // Blue
        { label: 'Precision', value: metrics.precision ?? 0, color: '#ff7f0e' }, // Orange
        { label: 'Recall', value: metrics.recall ?? 0, color: '#2ca02c' },     // Green
        { label: 'F1 Score', value: metrics.f1_score ?? 0, color: '#d62728' },   // Red
    ];

    const svgWidth = 450;
    const svgHeight = 200;
    const padding = 40;
    const gap = 35; // 막대 사이 간격
    const extraLeftGap = 30; // y축과 첫번째 막대 사이 간격
    const barShrink = 10; // 막대 너비를 줄이는 값
    const n = data.length;
    const totalGap = gap * (n - 1);
    const barWidth = ((svgWidth - 2 * padding - totalGap - extraLeftGap) / n) - barShrink;
    const graphOffsetX = 40; // 그래프 전체 x축 오프셋

    const yScale = (value: number) => svgHeight - padding - (value / 1.0) * (svgHeight - 2 * padding);

    // Y-axis ticks
    const yTicks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', p: 1, flexGrow: 1 }}>
            <Typography variant="subtitle2" gutterBottom sx={{ mb: 4, fontSize: '1.2em', width: '90%', textAlign: 'center', fontWeight: 'bold' }}>Overall Model Performance Metrics</Typography>
            <svg width={svgWidth} height={svgHeight} viewBox={`0 0 ${svgWidth} ${svgHeight}`} style={{ border: 'none', background: 'linear-gradient(135deg, #f8f8f8 80%, #e3f2fd 100%)', borderRadius: 16, boxShadow: '0 2px 12px #e3f2fd' }}>
                {/* Grid Lines (Y-axis only) */}
                {yTicks.map((tick, i) => (
                    <line key={`y-grid-${i}`} x1={padding + graphOffsetX} y1={yScale(tick)} x2={svgWidth - padding + graphOffsetX} y2={yScale(tick)} stroke="#eee" strokeWidth="1" />
                ))}

                {/* Y-axis */}
                <line x1={padding + graphOffsetX} y1={padding} x2={padding + graphOffsetX} y2={svgHeight - padding} stroke="black" strokeWidth="1" />
                {yTicks.map((tick, i) => (
                    <React.Fragment key={`y-tick-${i}`}>
                        <line x1={padding + graphOffsetX} y1={yScale(tick)} x2={padding - 5 + graphOffsetX} y2={yScale(tick)} stroke="black" strokeWidth="1" />
                        <text x={padding - 10 + graphOffsetX} y={yScale(tick) + 3} textAnchor="end" fontSize="10" fill="black">{tick.toFixed(1)}</text>
                    </React.Fragment>
                ))}
                <text x={-20 + graphOffsetX} y={svgHeight / 2} textAnchor="middle" transform={`rotate(-90 ${-20 + graphOffsetX} ${svgHeight / 2})`} fontSize="12" fill="black" fontWeight="bold">SCORE</text>

                {/* Bars and Labels */}
                {data.map((item, i) => {
                    const x = padding + extraLeftGap + i * (barWidth + gap) + graphOffsetX;
                    const y = yScale(item.value);
                    const barHeight = svgHeight - padding - y;
                    // 값에 따라 색상 변화
                    let barColor = item.color;
                    if (item.value >= 0.95) barColor = 'url(#barGreen)';
                    else if (item.value >= 0.7) barColor = 'url(#barYellow)';
                    else barColor = 'url(#barRed)';
                    return (
                        <React.Fragment key={item.label}>
                            <defs>
                                <linearGradient id="barGreen" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="0%" stopColor="#43e97b" />
                                    <stop offset="100%" stopColor="#38f9d7" />
                                </linearGradient>
                                <linearGradient id="barYellow" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="0%" stopColor="#fceabb" />
                                    <stop offset="100%" stopColor="#f8b500" />
                                </linearGradient>
                                <linearGradient id="barRed" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="0%" stopColor="#f85032" />
                                    <stop offset="100%" stopColor="#e73827" />
                                </linearGradient>
                            </defs>
                            <rect 
                                x={x}
                                y={y}
                                width={barWidth}
                                height={barHeight}
                                fill={barColor}
                                rx={8}
                                style={{ filter: 'drop-shadow(0 2px 6px #b2dfdb)' }}
                            />
                            <text x={x + barWidth / 2} y={y - 10} textAnchor="middle" fontSize="16" fontWeight="bold" fill="#1976d2">{item.value.toFixed(2)}</text>
                            <text x={x + barWidth / 2} y={svgHeight - padding + 15} textAnchor="middle" fontSize="12" fontWeight="bold" fill="#333">{item.label}</text>
                        </React.Fragment>
                    );
                })}
            </svg>
        </Box>
    );
};

export default MetricsBarChart; 