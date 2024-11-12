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
    width = "auto",
    textColor = "#000000",
}) => {
    return (
        <thead>
            <tr>
                {columns.map((column, index) => (
                    <th
                        key={index}
                        style={{ width, color: textColor }}
                        className={`text-left px-4 py-2`}
                    >
                        {column}
                    </th>
                ))}
            </tr>
        </thead>
    );
};

export default TableHeader;
