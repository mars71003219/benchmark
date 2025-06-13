import React from 'react';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer
} from 'recharts';

interface PRCurveChartProps {
    metricsHistory: { precision: number; recall: number; }[];
}

const PRCurveChart: React.FC<PRCurveChartProps> = ({ metricsHistory }) => {
    // Recharts는 데이터 배열을 필요로 합니다. metricsHistory는 이미 적절한 형식입니다.
    // 각 데이터 포인트는 { precision: P, recall: R } 형태를 가집니다.
    // x-축은 Recall, y-축은 Precision이 될 것입니다.

    return (
        <ResponsiveContainer width="100%" height="100%">
            <LineChart
                data={metricsHistory}
                margin={{
                    top: 5,
                    right: 30,
                    left: 20,
                    bottom: 5,
                }}
            >
                <CartesianGrid strokeDasharray="3 3" />
                {/* @ts-ignore */}
                <XAxis 
                    type="number" 
                    dataKey="recall" 
                    name="Recall" 
                    domain={[0, 1]} 
                />
                {/* @ts-ignore */}
                <YAxis 
                    type="number" 
                    dataKey="precision" 
                    name="Precision" 
                    domain={[0, 1]} 
                />
                {/* @ts-ignore */}
                <Tooltip />
                {/* @ts-ignore */}
                <Legend />
                {/* @ts-ignore */}
                <Line 
                    type="monotone" 
                    dataKey="precision" 
                    stroke="#8884d8" 
                    activeDot={{ r: 8 }} 
                    name="Precision" 
                />
                {/* @ts-ignore */}
                <Line 
                    type="monotone" 
                    dataKey="recall" 
                    stroke="#82ca9d" 
                    activeDot={{ r: 8 }} 
                    name="Recall" 
                />
            </LineChart>
        </ResponsiveContainer>
    );
};

export default PRCurveChart; 