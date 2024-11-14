import React from "react";

interface MemberTableHeaderProps {
  /** 컬럼에 입력되는 정보 */
  columns: (string | JSX.Element)[];
  /** 넓이(ex - 16px , 5rem) */
  width?: string;
  /** 텍스트 색상 */
  textColor?: string;
}

const MemberTableHeader: React.FC<MemberTableHeaderProps> = ({
  columns,
  width = "auto",
  textColor = "#4B5563",
}) => {
  return (
    <thead className="border-b border-gray-200 bg-white sticky top-0 z-10">
      <tr>
        {columns.map((column, index) => (
          <th
            key={index}
            className="pl-2 pr-6 py-3.5 text-left text-sm font-semibold tracking-wider whitespace-nowrap"
            style={{ color: textColor, width }}
          >
            {column}
          </th>
        ))}
      </tr>
    </thead>
  );
};

export default MemberTableHeader;
