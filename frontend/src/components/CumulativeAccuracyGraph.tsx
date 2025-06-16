import React from 'react';
import { Box, Typography, Paper } from '@mui/material';

interface CumulativeAccuracyGraphProps {
    cumulativeAccuracyHistory: { processed_clips: number; accuracy: number; }[];
}

const CumulativeAccuracyGraph: React.FC<CumulativeAccuracyGraphProps> = ({ cumulativeAccuracyHistory }) => {
    if (!cumulativeAccuracyHistory || cumulativeAccuracyHistory.length === 0) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', width: '100%' }}>
                <Typography variant="body2" color="text.secondary">누적 정확도 데이터 없음</Typography>
            </Box>
        );
    }

    // Get the last accuracy for the final accuracy line
    const finalAccuracy = cumulativeAccuracyHistory[cumulativeAccuracyHistory.length - 1]?.accuracy || 0;

    // Determine max values for scaling
    const maxProcessedClips = Math.max(...cumulativeAccuracyHistory.map(d => d.processed_clips), 0);
    const maxAccuracy = 1.0; // Accuracy is always between 0 and 1

    const svgWidth = 800;
    const svgHeight = 200;
    const padding = 30; // Padding for axes labels

    const xScale = (value: number) => (value / maxProcessedClips) * (svgWidth - 2 * padding) + padding;
    const yScale = (value: number) => svgHeight - padding - (value / maxAccuracy) * (svgHeight - 2 * padding);

    const points = cumulativeAccuracyHistory.map(d => 
        `${xScale(d.processed_clips)},${yScale(d.accuracy)}`
    ).join(' ');

    // X-axis ticks and labels
    const xTicks = [0, Math.floor(maxProcessedClips / 2), maxProcessedClips].filter((v, i, a) => a.indexOf(v) === i);
    const yTicks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', p: 1, flexGrow: 1 }}>
            <Typography variant="subtitle2" gutterBottom sx={{ mb: 1, fontSize: '1.2em', width: '90%', textAlign: 'center' }}>Cumulative Accuracy over Test Progress</Typography>
            <svg width={svgWidth} height={svgHeight} viewBox={`0 0 ${svgWidth} ${svgHeight}`} style={{ border: '1px solid #ccc', background: '#f8f8f8' }}>
                {/* Grid Lines (simplified) */}
                {xTicks.map((tick, i) => (
                    <line key={`x-grid-${i}`} x1={xScale(tick)} y1={padding} x2={xScale(tick)} y2={svgHeight - padding} stroke="#eee" strokeWidth="1" />
                ))}
                {yTicks.map((tick, i) => (
                    <line key={`y-grid-${i}`} x1={padding} y1={yScale(tick)} x2={svgWidth - padding} y2={yScale(tick)} stroke="#eee" strokeWidth="1" />
                ))}

                {/* X-axis */}
                <line x1={padding} y1={svgHeight - padding} x2={svgWidth - padding} y2={svgHeight - padding} stroke="black" strokeWidth="1" />
                {xTicks.map((tick, i) => (
                    <React.Fragment key={`x-tick-${i}`}>
                        <line x1={xScale(tick)} y1={svgHeight - padding} x2={xScale(tick)} y2={svgHeight - padding + 5} stroke="black" strokeWidth="1" />
                        <text x={xScale(tick)} y={svgHeight - padding + 15} textAnchor="middle" fontSize="10" fill="black">{tick}</text>
                    </React.Fragment>
                ))}
                <text x={svgWidth / 2} y={svgHeight - 5} textAnchor="middle" fontSize="12" fill="black">Number of Videos Processed</text>

                {/* Y-axis */}
                <line x1={padding} y1={padding} x2={padding} y2={svgHeight - padding} stroke="black" strokeWidth="1" />
                {yTicks.map((tick, i) => (
                    <React.Fragment key={`y-tick-${i}`}>
                        <line x1={padding} y1={yScale(tick)} x2={padding - 5} y2={yScale(tick)} stroke="black" strokeWidth="1" />
                        <text x={padding - 10} y={yScale(tick) + 3} textAnchor="end" fontSize="10" fill="black">{tick.toFixed(1)}</text>
                    </React.Fragment>
                ))}
                <text x={10} y={svgHeight / 2} textAnchor="middle" transform={`rotate(-90 ${10} ${svgHeight / 2})`} fontSize="12" fill="black">Cumulative Accuracy</text>

                {/* Data Line */}
                <polyline fill="none" stroke="blue" strokeWidth="2" points={points} />
                {cumulativeAccuracyHistory.map((d, i) => (
                    <circle key={`point-${i}`} cx={xScale(d.processed_clips)} cy={yScale(d.accuracy)} r="2" fill="blue" />
                ))}

            </svg>
        </Box>
    );
};

export default CumulativeAccuracyGraph; 