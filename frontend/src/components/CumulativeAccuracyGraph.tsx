import React from 'react';
import { Box, Typography, Paper } from '@mui/material';

interface CumulativeAccuracyGraphProps {
    cumulativeAccuracyHistory: { processed_clips: number; accuracy: number; }[];
}

const CumulativeAccuracyGraph: React.FC<CumulativeAccuracyGraphProps> = ({ cumulativeAccuracyHistory }) => {
    // 항상 최상단에서 호출!
    const [hoveredIdx, setHoveredIdx] = React.useState<number | null>(null);

    // Get the last accuracy for the final accuracy line
    const finalAccuracy = cumulativeAccuracyHistory && cumulativeAccuracyHistory.length > 0 ? cumulativeAccuracyHistory[cumulativeAccuracyHistory.length - 1]?.accuracy : 0;

    // Determine max values for scaling
    const maxProcessedClips = cumulativeAccuracyHistory && cumulativeAccuracyHistory.length > 0 ? Math.max(...cumulativeAccuracyHistory.map(d => d.processed_clips), 0) : 1;
    const maxAccuracy = 1.0; // Accuracy is always between 0 and 1

    const svgWidth = 450;
    const svgHeight = 200;
    const padding = 40; // Padding for axes labels
    const offset = 30;

    const xScale = (value: number) =>
        (value / maxProcessedClips) * (svgWidth - 2 * padding) + padding + offset;
    const yScale = (value: number) => svgHeight - padding - (value / maxAccuracy) * (svgHeight - 2 * padding);

    const points = cumulativeAccuracyHistory && cumulativeAccuracyHistory.length > 0 ? cumulativeAccuracyHistory.map(d => 
        `${xScale(d.processed_clips)},${yScale(d.accuracy)}`
    ).join(' ') : '';

    // X-axis ticks and labels
    const xTicks = [0, Math.floor(maxProcessedClips / 2), maxProcessedClips].filter((v, i, a) => a.indexOf(v) === i);
    const yTicks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', p: 1, flexGrow: 1 }}>
            <Typography variant="subtitle2" gutterBottom sx={{ mb: 1, fontSize: '1.2em', width: '90%', textAlign: 'center', fontWeight: 'bold' }}>Cumulative Accuracy over Test Progress</Typography>
            <svg width={svgWidth} height={svgHeight} viewBox={`0 0 ${svgWidth} ${svgHeight}`} style={{ border: 'none', background: 'linear-gradient(135deg, #f8f8f8 80%, #e3f2fd 100%)', borderRadius: 16, boxShadow: '0 2px 12px #e3f2fd' }}>
                {/* Grid Lines (simplified) */}
                {xTicks.map((tick, i) => (
                    <line key={`x-grid-${i}`} x1={xScale(tick)} y1={padding} x2={xScale(tick)} y2={svgHeight - padding} stroke="#eee" strokeWidth="1" />
                ))}
                {yTicks.map((tick, i) => (
                    <line key={`y-grid-${i}`} x1={padding} y1={yScale(tick)} x2={svgWidth - padding} y2={yScale(tick)} stroke="#eee" strokeWidth="1" />
                ))}

                {/* X-axis */}
                <line x1={padding + offset} y1={svgHeight - padding} x2={svgWidth - padding + offset} y2={svgHeight - padding} stroke="black" strokeWidth="1" />
                {xTicks.map((tick, i) => (
                    <React.Fragment key={`x-tick-${i}`}>
                        <line x1={xScale(tick)} y1={svgHeight - padding} x2={xScale(tick)} y2={svgHeight - padding + 5} stroke="black" strokeWidth="1" />
                        <text x={xScale(tick)} y={svgHeight - padding + 15} textAnchor="middle" fontSize="12" fill="black">{tick}</text>
                    </React.Fragment>
                ))}
                <text x={svgWidth / 2 + offset / 2} y={svgHeight - 5} textAnchor="middle" fontSize="14" fill="black" fontWeight="bold">Number of Videos Processed</text>

                {/* Y-axis */}
                <line x1={padding + offset} y1={padding} x2={padding + offset} y2={svgHeight - padding} stroke="black" strokeWidth="1" />
                {yTicks.map((tick, i) => (
                    <React.Fragment key={`y-tick-${i}`}>
                        <line x1={padding} y1={yScale(tick)} x2={padding - 5} y2={yScale(tick)} stroke="black" strokeWidth="1" />
                        <text x={padding + offset - 10} y={yScale(tick) + 3} textAnchor="end" fontSize="12" fill="black">{tick.toFixed(1)}</text>
                    </React.Fragment>
                ))}
                <text x={-20 + offset} y={svgHeight / 2} textAnchor="middle" transform={`rotate(-90 ${-20 + offset} ${svgHeight / 2})`} fontSize="14" fill="black" fontWeight="bold">Cumulative Accuracy</text>

                {/* Data Line (only if data exists) */}
                {points && (
                    <polyline fill="none" stroke="#1976d2" strokeWidth="3" points={points} style={{ filter: 'drop-shadow(0 2px 6px #b2dfdb)' }} />
                )}
                {/* Data Points (only if data exists) */}
                {cumulativeAccuracyHistory && cumulativeAccuracyHistory.length > 0 && cumulativeAccuracyHistory.map((d, i) => (
                    <g key={`point-${i}`}
                        onMouseEnter={() => setHoveredIdx(i)}
                        onMouseLeave={() => setHoveredIdx(null)}
                        style={{ cursor: 'pointer' }}>
                        <circle cx={xScale(d.processed_clips)} cy={yScale(d.accuracy)} r={hoveredIdx === i ? 7 : 4} fill={hoveredIdx === i ? '#43e97b' : '#1976d2'} style={{ transition: 'r 0.2s, fill 0.2s' }} />
                        {hoveredIdx === i && (
                            <g>
                                <rect x={xScale(d.processed_clips) - 30} y={yScale(d.accuracy) - 35} width="60" height="24" rx="6" fill="#fff" stroke="#1976d2" strokeWidth="1" />
                                <text x={xScale(d.processed_clips)} y={yScale(d.accuracy) - 20} textAnchor="middle" fontSize="13" fontWeight="bold" fill="#1976d2">{(d.accuracy * 100).toFixed(1)}%</text>
                            </g>
                        )}
                    </g>
                ))}
            </svg>
        </Box>
    );
};

export default CumulativeAccuracyGraph; 