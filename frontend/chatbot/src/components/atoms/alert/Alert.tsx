import React, { useEffect } from "react";
import Swal, { SweetAlertIcon } from "sweetalert2";

export interface AlertProps {
    icon?: SweetAlertIcon;
    title?: string;
    text?: string;
    showAlert: boolean;
    onClose?: () => void;
}

const Alert: React.FC<AlertProps> = ({
    icon = "success",
    title = "안녕하세요",
    text = "내용입니다.",
    showAlert,
    onClose,
}) => {
    useEffect(() => {
        if (showAlert) {
            Swal.fire({
                icon,
                title,
                text,
            }).then(() => {
                if (onClose) onClose();
            });
        }
    }, [showAlert, icon, title, text, onClose]);

    return null;
};

export default Alert;
