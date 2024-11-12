import React, { ReactNode } from "react";

interface ModalProps {
	children: ReactNode;
	onClose: () => void;
}

const Modal: React.FC<ModalProps> = ({ children, onClose }) => {
	return (
		<div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
			<div className="bg-white p-6 rounded-md w-80 max-w-full shadow-lg relative">
				<button
					onClick={onClose}
					className="absolute top-2 right-2 text-gray-600 hover:text-gray-800"
				>
					✕
				</button>
				{children}
			</div>
		</div>
	);
};

export default Modal;
