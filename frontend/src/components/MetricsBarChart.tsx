import React from 'react';
import { Box, Typography } from '@mui/material';

interface MetricsBarChartProps {
    metrics: {
        accuracy: number;
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

    const data = [
        { label: 'Accuracy', value: metrics.accuracy ?? 0, color: '#1f77b4' }, // Blue
        { label: 'Precision', value: metrics.precision ?? 0, color: '#ff7f0e' }, // Orange
        { label: 'Recall', value: metrics.recall ?? 0, color: '#2ca02c' },     // Green
        { label: 'F1 Score', value: metrics.f1_score ?? 0, color: '#d62728' },   // Red
    ];

    const svgWidth = 500;
    const svgHeight = 200;
    const padding = 30;
    const barWidth = (svgWidth - 2 * padding) / data.length - 10; // Reduced bar width for spacing

    const yScale = (value: number) => svgHeight - padding - (value / 1.0) * (svgHeight - 2 * padding);

    // Y-axis ticks
    const yTicks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', p: 1, flexGrow: 1 }}>
            <Typography variant="subtitle2" gutterBottom>Overall Model Performance Metrics</Typography>
            <svg width={svgWidth} height={svgHeight} viewBox={`0 0 ${svgWidth} ${svgHeight}`} style={{ border: '1px solid #ccc', background: '#f8f8f8' }}>
                {/* Grid Lines (Y-axis only) */}
                {yTicks.map((tick, i) => (
                    <line key={`y-grid-${i}`} x1={padding} y1={yScale(tick)} x2={svgWidth - padding} y2={yScale(tick)} stroke="#eee" strokeWidth="1" />
                ))}

                {/* Y-axis */}
                <line x1={padding} y1={padding} x2={padding} y2={svgHeight - padding} stroke="black" strokeWidth="1" />
                {yTicks.map((tick, i) => (
                    <React.Fragment key={`y-tick-${i}`}>
                        <line x1={padding} y1={yScale(tick)} x2={padding - 5} y2={yScale(tick)} stroke="black" strokeWidth="1" />
                        <text x={padding - 10} y={yScale(tick) + 3} textAnchor="end" fontSize="10" fill="black">{tick.toFixed(1)}</text>
                    </React.Fragment>
                ))}
                <text x={10} y={svgHeight / 2} textAnchor="middle" transform={`rotate(-90 ${10} ${svgHeight / 2})`} fontSize="12" fill="black">Score</text>

                {/* Bars and Labels */}
                {data.map((item, i) => {
                    const x = padding + i * (barWidth + 10) + 5; // +5 for minor centering
                    const y = yScale(item.value);
                    const barHeight = svgHeight - padding - y;
                    return (
                        <React.Fragment key={item.label}>
                            <rect 
                                x={x}
                                y={y}
                                width={barWidth}
                                height={barHeight}
                                fill={item.color}
                            />
                            <text x={x + barWidth / 2} y={y - 5} textAnchor="middle" fontSize="10" fill="black">{item.value.toFixed(3)}</text>
                            <text x={x + barWidth / 2} y={svgHeight - padding + 15} textAnchor="middle" fontSize="10" fill="black">{item.label}</text>
                        </React.Fragment>
                    );
                })}
            </svg>
        </Box>
    );
};

export default MetricsBarChart; 