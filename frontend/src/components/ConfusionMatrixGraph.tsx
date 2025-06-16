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
        width: '250px',
        height: '250px',
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
        fontSize: '1.2em',
        padding: '2px',
    };

    const labelStyle: React.CSSProperties = {
        fontSize: '0.6em',
        fontWeight: 'normal',
        color: '#555',
        marginTop: '1px',
        textAlign: 'center',
        lineHeight: '1',
    };

    const correctColor = '#1f77b4';
    const incorrectColor = '#a6cee3';

    const leftLabelWidth = 70;

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', p: 1.5 }}>
            <Typography variant="subtitle2" gutterBottom sx={{ mb: 1.0, fontSize: '1.2em', width: '110%', textAlign: 'center' }}>Confusion Matrix</Typography>

            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', width: 'fit-content' }}>

                <Box sx={{
                    display: 'flex',
                    width: `calc(${leftLabelWidth}px + ${matrixStyle.width})`,
                    justifyContent: 'center',
                    mb: 0.5
                }}>
                    <Box sx={{ width: `${leftLabelWidth}px` }} />
                    <Typography variant="caption" sx={{ fontSize: '0.7em', flexGrow: 1, textAlign: 'center' }}>Predicted Label</Typography>
                </Box>

                <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                    <Box sx={{ width: `${leftLabelWidth}px` }} />
                    <Box sx={{ display: 'flex', width: matrixStyle.width }}>
                        <Box sx={{ width: '50%', textAlign: 'center' }}>
                            <Typography variant="caption" sx={{ fontWeight: 'medium', fontSize: '0.7em' }}>NonFight</Typography>
                        </Box>
                        <Box sx={{ width: '50%', textAlign: 'center' }}>
                            <Typography variant="caption" sx={{ fontWeight: 'medium', fontSize: '0.7em' }}>Fight</Typography>
                        </Box>
                    </Box>
                </Box>

                <Box sx={{ display: 'flex', flexDirection: 'row', alignItems: 'center' }}>
                    <Box sx={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        height: matrixStyle.height, 
                        mr: -1,
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

                    <div style={matrixStyle}>
                        <div style={{ ...cellStyle, backgroundColor: correctColor }}>
                            <Typography variant="inherit" sx={{ color: '#fff', fontSize: '1.2em', fontWeight: 'bold' }}>
                                {tn}
                            </Typography>
                            <span style={{ ...labelStyle, color: '#fff' }}>TN<br/>(NonFight)</span>
                        </div>
                        <div style={{ ...cellStyle, backgroundColor: incorrectColor }}>
                            <Typography variant="inherit" sx={{ color: '#000', fontSize: '1.2em', fontWeight: 'bold' }}>
                                {fp}
                            </Typography>
                            <span style={{ ...labelStyle, color: '#000' }}>FP<br/>(NonFight)</span>
                        </div>
                        <div style={{ ...cellStyle, backgroundColor: incorrectColor }}>
                            <Typography variant="inherit" sx={{ color: '#000', fontSize: '1.2em', fontWeight: 'bold' }}>
                                {fn}
                            </Typography>
                            <span style={{ ...labelStyle, color: '#000' }}>FN<br/>(Fight)</span>
                        </div>
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