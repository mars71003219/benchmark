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
        width: '200px', // Fixed width for the matrix
        height: '200px', // Fixed height for the matrix
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
    };

    const labelStyle: React.CSSProperties = {
        fontSize: '0.8em',
        fontWeight: 'normal',
        color: '#555',
        marginTop: '4px',
    };

    // Define colors based on the provided image (darker blue for correct, lighter blue for incorrect)
    const correctColor = '#1f77b4'; // Dark blue (similar to TN/TP in image)
    const incorrectColor = '#a6cee3'; // Lighter blue (similar to FP/FN in image)

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', p: 2 }}>
            <Typography variant="subtitle2" gutterBottom>Confusion Matrix</Typography>

            {/* Container for everything below the main title */}
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: 'fit-content' }}>

                {/* Predicted Label (axis header) and column headers */}
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mb: 1 }}>
                    <Typography variant="caption">Predicted Label</Typography>
                    <Box sx={{ display: 'flex', justifyContent: 'space-around', width: matrixStyle.width, mt: 0.5 }}>
                        <Typography variant="caption" sx={{ flex: 1, textAlign: 'center' }}>NonFight</Typography>
                        <Typography variant="caption" sx={{ flex: 1, textAlign: 'center' }}>Fight</Typography>
                    </Box>
                </Box>

                {/* Actual Label (axis header) and Matrix with row headers */}
                <Box sx={{ display: 'flex', flexDirection: 'row', alignItems: 'center'}}>
                    {/* Actual Label (axis header) - centered vertically */}
                    <Box sx={{ display: 'flex', alignItems: 'center', height: matrixStyle.height, mr: 0.5 }}>
                        <Typography variant="caption" sx={{ transform: 'rotate(-90deg)', whiteSpace: 'nowrap' }}>Actual Label</Typography>
                    </Box>

                    {/* Row headers (NonFight, Fight) */}
                    <Box sx={{ display: 'flex', flexDirection: 'column', mr: 1, height: matrixStyle.height, justifyContent: 'space-around', alignItems: 'center' }}>
                        <Typography variant="caption" sx={{ transform: 'rotate(-90deg)', whiteSpace: 'nowrap' }}>NonFight</Typography>
                        <Typography variant="caption" sx={{ transform: 'rotate(-90deg)', whiteSpace: 'nowrap' }}>Fight</Typography>
                    </Box>

                    {/* Confusion Matrix Grid */}
                    <div style={matrixStyle}>
                        {/* TN */}
                        <div style={{ ...cellStyle, backgroundColor: correctColor }}>
                            {tn}
                            <span style={labelStyle}>TN (NonFight)</span>
                        </div>
                        {/* FP */}
                        <div style={{ ...cellStyle, backgroundColor: incorrectColor }}>
                            {fp}
                            <span style={labelStyle}>FP (NonFight)</span>
                        </div>
                        {/* FN */}
                        <div style={{ ...cellStyle, backgroundColor: incorrectColor }}>
                            {fn}
                            <span style={labelStyle}>FN (Fight)</span>
                        </div>
                        {/* TP */}
                        <div style={{ ...cellStyle, backgroundColor: correctColor }}>
                            {tp}
                            <span style={labelStyle}>TP (Fight)</span>
                        </div>
                    </div>
                </Box>
            </Box>
        </Box>
    );
};

export default ConfusionMatrixGraph; 