import React, { useEffect } from "react";

interface ModalProps {
  children: React.ReactNode;
  onClose: () => void;
  title?: string;
}

const Modal: React.FC<ModalProps> = ({ children, onClose, title }) => {
  useEffect(() => {
    const handleEsc = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handleEsc);
    document.body.style.overflow = "hidden";

    return () => {
      window.removeEventListener("keydown", handleEsc);
      document.body.style.overflow = "unset";
    };
  }, [onClose]);

  const handleBackdropClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 overflow-auto bg-black bg-opacity-40 backdrop-blur-sm flex items-center justify-center p-4
                 animate-[fadeIn_0.2s_ease-out]"
      onClick={handleBackdropClick}
    >
      <div
        className="bg-white rounded-xl shadow-[0_0_40px_rgba(0,0,0,0.1)] 
                   max-w-md w-full h-50vh overflow-auto
                   animate-[slideIn_0.3s_ease-out]
                   border border-gray-100"
      >
        {/* 모달 컨텐츠 */}
        <div className="px-6 py-5">{children}</div>
      </div>
    </div>
  );
};

export default Modal;
