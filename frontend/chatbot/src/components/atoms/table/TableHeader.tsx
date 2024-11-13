import React from "react";

interface TableHeaderProps {
  /** 컬럼에 입력되는 정보 */
  columns: (string | JSX.Element)[];
  /** 넓이(ex - 16px , 5rem) */
  width?: string;
  /** 텍스트 색상 */
  textColor?: string;
}

const TableHeader: React.FC<TableHeaderProps> = ({
  columns,
  textColor = "#4B5563",
}) => {
  return (
    <thead className="border-b border-gray-200">
      <tr>
        {/* 즐겨찾기 아이콘 열 */}
        <th
          className="w-16 px-6 py-3.5 text-left text-sm font-semibold tracking-wider"
          style={{ color: textColor }}
        >
          {columns[0]}
        </th>
        {/* 작성자 열 */}
        <th
          className="w-28 px-6 py-3.5 text-left text-sm font-semibold tracking-wider whitespace-nowrap"
          style={{ color: textColor }}
        >
          {columns[1]}
        </th>
        {/* 제목 열 - 남은 공간 모두 차지 */}
        <th
          className="px-6 py-3.5 text-left text-sm font-semibold tracking-wider"
          style={{ color: textColor }}
        >
          {columns[2]}
        </th>
        {/* 등록일 열 */}
        <th
          className="w-32 px-6 py-3.5 text-left text-sm font-semibold tracking-wider whitespace-nowrap"
          style={{ color: textColor }}
        >
          {columns[3]}
        </th>
      </tr>
    </thead>
  );
};

export default TableHeader;
