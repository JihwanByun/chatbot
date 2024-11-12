import { useState, useEffect } from 'react';

export interface CardWithModalProps {
  title?: string;
  content: string;
}

const Card = ({ title, content }: CardWithModalProps) => {
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [isOpen]);

  const handleBackdropClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) {
      setIsOpen(false);
    }
  };

  return (
    <>
      <div
        onClick={() => setIsOpen(true)}
        className="relative p-4 border border-gray-200 bg-white rounded-2xl shadow-sm max-w-full hover:bg-gray-100 hover:cursor-pointer transition-all duration-100 ease-in-out"
      >
        {title && <p className="text-sm mb-2 truncate font-bold text-gray-800 max-w-fit">{title}</p>}
        <p className="text-sm mb-2 line-clamp-2 lg:line-clamp-3 text-gray-600">{content}</p>
      </div>

      {isOpen && (
        <div 
          onClick={handleBackdropClick}
          className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4 animate-in fade-in duration-200"
        >
          <div className="relative bg-white rounded-2xl max-w-2xl w-full mx-4 p-8 shadow-2xl animate-in slide-in-from-bottom-8 duration-300">
            <div className="flex justify-between items-start mb-6">
              {title && <h2 className="text-2xl font-bold text-gray-800">{title}</h2>}
              <button 
                onClick={() => setIsOpen(false)}
                className="text-gray-400 hover:text-gray-600 transition-colors duration-200 p-1 hover:bg-gray-100 rounded-full absolute right-4 top-4"
              >
                <svg 
                  className="w-6 h-6" 
                  fill="none" 
                  stroke="currentColor" 
                  viewBox="0 0 24 24"
                >
                  <path 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                    strokeWidth="2" 
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </div>
            <div className="mt-4 max-h-[60vh] overflow-y-auto">
              <p className="text-gray-600 leading-relaxed whitespace-pre-wrap text-base">{content}</p>
            </div>
            <div className="mt-8 flex justify-end">
              <button
                onClick={() => setIsOpen(false)}
                className="px-6 py-2 bg-gray-50 text-gray-700 rounded-lg hover:bg-gray-100 transition-colors duration-200 font-medium focus:outline-none focus:ring-2 focus:ring-gray-200 active:bg-gray-200"
              >
                닫기
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default Card;