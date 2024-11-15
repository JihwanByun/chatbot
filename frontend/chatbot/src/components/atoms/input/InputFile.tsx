interface FileUploadFormProps {
  selectedFile: File | null;
  setSelectedFile: (file: File | null) => void;
  handleFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

const InputFile: React.FC<FileUploadFormProps> = ({
  selectedFile,
  setSelectedFile,
  handleFileChange,
}) => {
  const resetFile = () => {
    setSelectedFile(null);
    const fileInput = document.querySelector(
      'input[type="file"]'
    ) as HTMLInputElement;
    if (fileInput) fileInput.value = "";
  };

  return (
    <div className="p-4 border rounded-md">
      {!selectedFile ? (
        // 파일이 선택되지 않았을 때
        <input
          type="file"
          accept=".pdf"
          onChange={handleFileChange}
          className="block w-full text-sm text-gray-500
            file:mr-4 file:py-2 file:px-4
            file:rounded-md file:border-0
            file:text-sm file:font-semibold
            file:bg-blue-50 file:text-blue-700
            hover:file:bg-blue-100"
        />
      ) : (
        // 파일이 선택되었을 때
        <div className="flex items-center justify-between text-sm text-gray-600 bg-gray-50 p-2 rounded">
          <div className="flex items-center flex-grow">
            <svg
              className="w-4 h-4 mr-2 text-gray-500"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"
              />
            </svg>
            <span className="truncate">{selectedFile.name}</span>
          </div>
          <div className="flex items-center ml-4">
            <label
              className="cursor-pointer text-blue-600 hover:text-blue-700 mr-4"
              htmlFor="fileInput"
            >
              변경
            </label>
            <input
              id="fileInput"
              type="file"
              accept=".pdf"
              onChange={handleFileChange}
              className="hidden"
            />
            <button
              onClick={resetFile}
              className="text-gray-500 hover:text-red-500 focus:outline-none"
              type="button"
              title="파일 삭제"
            >
              삭제
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default InputFile;