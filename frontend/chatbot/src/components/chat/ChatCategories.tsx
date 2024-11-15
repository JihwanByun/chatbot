import React, { useEffect, useState } from 'react';

export interface ChatCategory {
  name: string;
  prompt: string;
}

interface ChatCategoriesProps {
  categories: ChatCategory[];
  onCategoryClick: (category : ChatCategory)=>void
}

const ChatCategories: React.FC<ChatCategoriesProps> = ({ 
  categories,
  onCategoryClick
}) => {
  const [showAll, setShowAll] = useState(false); // 카테고리 더보기 상태
  const [initCategorySize, setInitCategorySize] = useState(5); // 기본 카테고리 개수

  // 화면 크기에 따라 INIT_CATEGORY_SIZE를 변경하는 함수
  const updateCategorySize = () => {
    if (window.innerWidth >= 1024) {
      setInitCategorySize(5); // lg 이상일 때
    } else if (window.innerWidth >= 768) {
      setInitCategorySize(3); // md 이상일 때
    } else {
      setInitCategorySize(2); // sm 이하일 때
    }
  };

  useEffect(() => {
    // 컴포넌트가 마운트될 때 초기 사이즈 설정
    updateCategorySize();
    
    // 윈도우 크기 변경 감지하여 사이즈 업데이트
    window.addEventListener('resize', updateCategorySize);
    
    return () => {
      window.removeEventListener('resize', updateCategorySize);
    };
  }, []);

  // 카테고리 목록을 보여주는 로직
  const displayedCategories = showAll ? categories : categories.slice(0, initCategorySize);

  return (
    <div className='flex justify-center items-center'>
      <div className={`grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-2`}>
        {displayedCategories.map((category, index) => (
          <button
            key={index}
            className="relative h-full px-5 py-2.5 rounded-lg
              border border-semesBlue/30
              text-sm font-semibold text-semesBlue
              transition-all duration-300 ease-in-out
              hover:border-semesBlue/60
              hover:scale-105 hover:-translate-y-0.5
              active:scale-95
              focus:outline-0 focus:ring-2 focus:ring-semesBlue/60
              shadow-sm
              overflow-hidden
              min-w-[100px] max-w-[200px]
              group"
            onClick={() => onCategoryClick(category)}
          >
            {/* Button Content */}
            <div className="relative flex items-center gap-2">
              <span className="text-semesBlue/70 group-hover:text-semesBlue transition-colors">
                #
              </span>
              <span className="truncate">{category.name}</span>
            </div>
          </button>
        ))}
      </div>
      {/* 더보기 버튼 표시 */}
      {(categories.length > initCategorySize && !showAll) && (
        <img 
          src="/src/assets/icons/dots-horizontal.svg" 
          onClick={() => setShowAll(prev => !prev)} 
          className="ml-3 cursor-pointer" 
          alt="Show more categories"
        />
      )}
    </div>
  );
};

export default ChatCategories;