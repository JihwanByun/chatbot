export interface Category {
  value: string;
  label: string;
}

export interface FormData {
  title: string;
  category: string;
  level: number;
}

export interface BoardCreateFormProps {
  formData?: FormData;
  categories?: Category[];
  onInputChange?: (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => void;
}

const defaultFormData: FormData = {
  title: '',
  category: '',
  level: 1
};

const defaultCategories: Category[] = [
  { value: 'general', label: '일반' },
  { value: 'tech', label: '기술' },
  { value: 'management', label: '경영' },
  { value: 'hr', label: '인사' }
];

const BoardCreateForm: React.FC<BoardCreateFormProps> = ({ 
  formData = defaultFormData, 
  categories = defaultCategories, 
  onInputChange = () => {}, 
}) => {

  return (
    <div className="py-6 bg-white border-b text-base">
      <div className="max-w-2xl space-y-6">
        {/* Title Input */}
        <div className="space-y-2">
          <label htmlFor="title" className="block font-medium text-gray-700">
            제목
          </label>
          <input
            type="text"
            id="title"
            name="title"
            value={formData.title}
            onChange={onInputChange}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            placeholder="제목을 입력하세요"
          />
        </div>

        {/* Category and Level Row */}
        <div className="flex space-x-6">
          {/* Category Dropdown */}
          <div className="space-y-2 w-1/2">
            <label htmlFor="category" className="block font-medium text-gray-700">
              카테고리
            </label>
            <select
              id="category"
              name="category"
              defaultValue={categories[0]?.value}  // 첫 번째 카테고리를 기본값으로
              value={formData.category}
              onChange={onInputChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              {categories.map((category) => (
                <option key={category.value} value={category.value}>
                  {category.label}
                </option>
              ))}
            </select>
          </div>

          {/* Level Radio Buttons */}
          <div className="space-y-2 w-1/2">
            <label className="block font-medium text-gray-700">레벨</label>
            <div className="flex space-x-6 pt-2">
              {[1, 2, 3].map((level) => (
                <label key={level} className="inline-flex items-center">
                  <input
                    type="radio"
                    name="level"
                    value={level}
                    checked={Number(formData.level) === level}
                    onChange={onInputChange}
                    className="h-4 w-4 text-blue-600 border-gray-300 focus:ring-blue-500"
                  />
                  <span className="ml-2 text-sm text-gray-700">Level {level}</span>
                </label>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BoardCreateForm;