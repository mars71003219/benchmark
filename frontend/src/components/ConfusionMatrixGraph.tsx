import React from 'react';
import { Box, Typography, Paper } from '@mui/material';

interface ConfusionMatrixGraphProps {
    metrics: {
        tp: number;
        tn: number;
        fp: number;
        fn: number;
    } | undefined;
}

const ConfusionMatrixGraph: React.FC<ConfusionMatrixGraphProps> = ({ metrics }) => {
    if (!metrics) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', width: '100%' }}>
                <Typography variant="body2" color="text.secondary">메트릭스 데이터 없음</Typography>
            </Box>
        );
    }

    const { tp, tn, fp, fn } = metrics;

    const matrixStyle: React.CSSProperties = {
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gridTemplateRows: '1fr 1fr',
        width: '160px', // 크기 줄임
        height: '160px', // 크기 줄임
        border: '1px solid #ccc',
        borderRadius: '4px',
        overflow: 'hidden',
    };

    const cellStyle: React.CSSProperties = {
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        border: '1px solid #eee',
        fontWeight: 'bold',
        fontSize: '1.2em', // 폰트 크기 줄임
        padding: '2px', // 패딩 줄임
    };

    const labelStyle: React.CSSProperties = {
        fontSize: '0.6em', // 폰트 크기 줄임
        fontWeight: 'normal',
        color: '#555',
        marginTop: '1px',
        textAlign: 'center',
        lineHeight: '1',
    };

    // Define colors based on the provided image
    const correctColor = '#1f77b4'; // Dark blue for correct predictions
    const incorrectColor = '#a6cee3'; // Light blue for incorrect predictions

    // 좌측 라벨 영역 너비 계산
    const leftLabelWidth = 70; // Actual Label + Row Headers 총 너비

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', p: 1.5 }}>
            <Typography variant="subtitle2" gutterBottom sx={{ mb: 1.5, fontSize: '0.9em' }}>Confusion Matrix</Typography>

            {/* Main container for the matrix and labels */}
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', width: 'fit-content' }}>

                {/* Predicted Label and Column Headers */}
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', mb: 0.5 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                        {/* 좌측 여백 */}
                        <Box sx={{ width: `${leftLabelWidth}px` }}>
                            <Typography variant="caption" sx={{ fontSize: '0.7em' }}>Predicted Label</Typography>
                        </Box>
                        {/* 컬럼 헤더들 */}
                        <Box sx={{ display: 'flex', width: matrixStyle.width }}>
                            <Box sx={{ width: '50%', textAlign: 'center' }}>
                                <Typography variant="caption" sx={{ fontWeight: 'medium', fontSize: '0.7em' }}>NonFight</Typography>
                            </Box>
                            <Box sx={{ width: '50%', textAlign: 'center' }}>
                                <Typography variant="caption" sx={{ fontWeight: 'medium', fontSize: '0.7em' }}>Fight</Typography>
                            </Box>
                        </Box>
                    </Box>
                </Box>

                {/* Actual Label and Matrix with Row Headers */}
                <Box sx={{ display: 'flex', flexDirection: 'row', alignItems: 'center' }}>
                    {/* Actual Label Axis Header */}
                    <Box sx={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        height: matrixStyle.height, 
                        mr: 0.5,
                        width: '35px',
                        justifyContent: 'center'
                    }}>
                        <Typography 
                            variant="caption" 
                            sx={{ 
                                transform: 'rotate(-90deg)', 
                                whiteSpace: 'nowrap',
                                transformOrigin: 'center center',
                                fontSize: '0.7em'
                            }}
                        >
                            Actual Label
                        </Typography>
                    </Box>

                    {/* Row Headers (NonFight, Fight) */}
                    <Box sx={{ 
                        display: 'flex', 
                        flexDirection: 'column', 
                        mr: 0.5,
                        height: matrixStyle.height, 
                        justifyContent: 'space-around', 
                        alignItems: 'center',
                        width: '35px'
                    }}>
                        <Typography 
                            variant="caption" 
                            sx={{ 
                                transform: 'rotate(-90deg)', 
                                whiteSpace: 'nowrap',
                                fontWeight: 'medium',
                                transformOrigin: 'center center',
                                fontSize: '0.7em'
                            }}
                        >
                            NonFight
                        </Typography>
                        <Typography 
                            variant="caption" 
                            sx={{ 
                                transform: 'rotate(-90deg)', 
                                whiteSpace: 'nowrap',
                                fontWeight: 'medium',
                                transformOrigin: 'center center',
                                fontSize: '0.7em'
                            }}
                        >
                            Fight
                        </Typography>
                    </Box>

                    {/* Confusion Matrix Grid */}
                    <div style={matrixStyle}>
                        {/* TN - Top Left */} 
                        <div style={{ ...cellStyle, backgroundColor: correctColor }}>
                            <Typography variant="inherit" sx={{ color: '#fff', fontSize: '1.2em', fontWeight: 'bold' }}>
                                {tn}
                            </Typography>
                            <span style={{ ...labelStyle, color: '#fff' }}>TN<br/>(NonFight)</span>
                        </div>
                        {/* FP - Top Right */} 
                        <div style={{ ...cellStyle, backgroundColor: incorrectColor }}>
                            <Typography variant="inherit" sx={{ color: '#000', fontSize: '1.2em', fontWeight: 'bold' }}>
                                {fp}
                            </Typography>
                            <span style={{ ...labelStyle, color: '#000' }}>FP<br/>(NonFight)</span>
                        </div>
                        {/* FN - Bottom Left */} 
                        <div style={{ ...cellStyle, backgroundColor: incorrectColor }}>
                            <Typography variant="inherit" sx={{ color: '#000', fontSize: '1.2em', fontWeight: 'bold' }}>
                                {fn}
                            </Typography>
                            <span style={{ ...labelStyle, color: '#000' }}>FN<br/>(Fight)</span>
                        </div>
                        {/* TP - Bottom Right */} 
                        <div style={{ ...cellStyle, backgroundColor: correctColor }}>
                            <Typography variant="inherit" sx={{ color: '#fff', fontSize: '1.2em', fontWeight: 'bold' }}>
                                {tp}
                            </Typography>
                            <span style={{ ...labelStyle, color: '#fff' }}>TP<br/>(Fight)</span>
                        </div>
                    </div>
                </Box>
            </Box>
        </Box>
    );
};

export default ConfusionMatrixGraph;