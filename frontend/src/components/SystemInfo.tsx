// frontend/src/components/SystemInfo.tsx

import React, { useEffect, useState } from 'react';
import { Box, Typography, Paper, Grid, LinearProgress, Stack } from '@mui/material';
import { SxProps } from '@mui/system';

// 백엔드 API 응답 타입 정의
interface SysInfo {
  cpu: number | null;
  ram: number | null;
  ram_used_mb: number | null;
  ram_total_mb: number | null;
  gpu: string | null;
  gpuMem: string | null;
  gpuUtil: number | null;
}

// SystemInfo 컴포넌트의 props 타입 정의
interface SystemInfoProps {
  sx?: SxProps;
}

// 개별 정보 항목을 위한 스타일 컴포넌트
const InfoCard = ({ title, value, unit, progress, color, secondaryText }: { 
    title: string, 
    value: number | string, 
    unit: string, 
    progress: number, 
    color: string,
    secondaryText?: string 
}) => {
    return (
        <Paper variant="outlined" sx={{ p: 1.5, textAlign: 'center', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <Typography variant="body2" sx={{ fontWeight: 'bold', color: 'text.secondary' }}>{title}</Typography>
            <Typography variant="h6">
                {value}<span style={{ fontSize: '0.8rem' }}>{unit}</span>
            </Typography>
            <LinearProgress
                variant="determinate"
                value={progress}
                sx={{
                    height: 6,
                    borderRadius: 3,
                    my: 0.5,
                    '& .MuiLinearProgress-bar': {
                        backgroundColor: color,
                    }
                }}
            />
            {secondaryText && (
                <Typography variant="caption" sx={{ color: 'text.secondary', mt: 0.5 }}>
                    {secondaryText}
                </Typography>
            )}
        </Paper>
    );
};


const SystemInfo: React.FC<SystemInfoProps> = ({ sx }) => {
  const [info, setInfo] = useState<SysInfo>({
    cpu: null, ram: null, ram_used_mb: null, ram_total_mb: null,
    gpu: null, gpuMem: null, gpuUtil: null
  });

  const parseGpuMem = (memString: string | null): [number, number, number] => {
    if (!memString) return [0, 0, 0];
    const matches = memString.match(/(\d+\.?\d*)\s*MB\s*\/\s*(\d+\.?\d*)\s*MB/);
    if (matches && matches.length === 3) {
      const used = parseFloat(matches[1]);
      const total = parseFloat(matches[2]);
      if (total > 0) {
        const percentage = (used / total) * 100;
        return [used, total, Math.round(percentage)];
      }
    }
    return [0, 0, 0];
  };

  useEffect(() => {
    const fetchInfo = async () => {
      try {
        const res = await fetch('/system_info');
        if (res.ok) {
          const data = await res.json();
          setInfo(data);
        }
      } catch (error) {
          console.error("시스템 정보 fetching 실패:", error);
      }
    };
    
    fetchInfo();
    const timer = setInterval(fetchInfo, 2000);
    return () => clearInterval(timer);
  }, []);
  
  const [gpuMemUsed, gpuMemTotal, gpuMemPercentage] = parseGpuMem(info.gpuMem);

  return (
    <Paper sx={{ p: 2, ...sx }}>
        <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold', textAlign: 'center' }}>
            시스템 모니터링
        </Typography>
        <Typography variant="body2" sx={{ textAlign: 'center', color: 'text.secondary', mb: 2 }}>
            {info.gpu || 'GPU 정보를 불러오는 중...'}
        </Typography>
      <Grid container spacing={2}>
        <Grid item xs={6}>
            <Stack spacing={2}>
                <InfoCard
                    title="CPU"
                    value={info.cpu?.toFixed(1) ?? '-'}
                    unit="%"
                    progress={info.cpu ?? 0}
                    color="#4dabf5"
                />
                <InfoCard
                    title="RAM"
                    value={info.ram?.toFixed(1) ?? '-'}
                    unit="%"
                    progress={info.ram ?? 0}
                    color="#69db7c"
                    secondaryText={info.ram_used_mb ? `${info.ram_used_mb.toFixed(0)} / ${info.ram_total_mb?.toFixed(0)} MB` : ''}
                />
            </Stack>
        </Grid>
        <Grid item xs={6}>
            <Stack spacing={2}>
                {/* GPU UTIL과 GPU MEM의 위치를 변경 */}
                <InfoCard
                    title="GPU UTIL"
                    value={info.gpuUtil ?? '-'}
                    unit="%"
                    progress={info.gpuUtil ?? 0}
                    color="#f06595"
                />
                <InfoCard
                    title="GPU MEM"
                    value={gpuMemPercentage}
                    unit="%"
                    progress={gpuMemPercentage}
                    color="#ffc078"
                    secondaryText={gpuMemTotal > 0 ? `${gpuMemUsed.toFixed(0)} / ${gpuMemTotal.toFixed(0)} MB` : ''}
                />
            </Stack>
        </Grid>
      </Grid>
    </Paper>
  );
};

export default SystemInfo;