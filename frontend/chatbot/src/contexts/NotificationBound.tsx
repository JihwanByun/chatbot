import { useAppSelector } from '@app/hooks';
import { useNotification } from '@contexts/NotificationContext';
import { useEffect } from 'react';

const NotificationBound : React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { connect, disconnect } = useNotification();
  const {name, employeeNum} = useAppSelector(state => state.users)

  const createUserId = () => {
    return name+"_"+employeeNum;
  }

  useEffect(() => {
    // 컴포넌트 마운트 시 연결
    connect(createUserId());

    // 컴포넌트 언마운트 시 연결 종료
    return () => {
      disconnect();
    };
  }, []);

  return (
    <div>
      {children}
    </div>
  );
};
export default NotificationBound;