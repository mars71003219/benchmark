import React from 'react';
import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Paper
} from '@mui/material';
import {
  CheckCircle as SuccessIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  VideoFile as VideoIcon
} from '@mui/icons-material';

interface Event {
  type: string;
  timestamp: string;
  video?: string;
  message?: string;
  results?: number;
  total_results?: number;
}

interface EventLogProps {
  events: Event[];
}

const EventLog: React.FC<EventLogProps> = ({ events }) => {
  const getEventIcon = (type: string) => {
    switch (type) {
      case 'video_processed':
        return <SuccessIcon color="success" />;
      case 'error':
        return <ErrorIcon color="error" />;
      case 'complete':
        return <InfoIcon color="info" />;
      default:
        return <VideoIcon />;
    }
  };

  const getEventText = (event: Event) => {
    switch (event.type) {
      case 'video_processed':
        return `${event.video} 처리 완료 (${event.results}개 결과)`;
      case 'error':
        return `오류 발생: ${event.message}`;
      case 'complete':
        return `모든 처리 완료 (총 ${event.total_results}개 결과)`;
      default:
        return event.message || '알 수 없는 이벤트';
    }
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        이벤트 로그
      </Typography>

      <Paper sx={{ maxHeight: 400, overflow: 'auto' }}>
        <List>
          {events.map((event, index) => (
            <ListItem key={index} divider>
              <ListItemIcon>
                {getEventIcon(event.type)}
              </ListItemIcon>
              <ListItemText
                primary={getEventText(event)}
                secondary={new Date(event.timestamp).toLocaleString()}
              />
            </ListItem>
          ))}
          
          {events.length === 0 && (
            <ListItem>
              <ListItemText
                primary="이벤트가 없습니다"
                secondary="비디오 처리를 시작하면 이벤트가 여기에 표시됩니다"
              />
            </ListItem>
          )}
        </List>
      </Paper>
    </Box>
  );
};

export default EventLog; 