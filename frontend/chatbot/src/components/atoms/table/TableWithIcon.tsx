import React from "react";
import TableHeader from "./TableHeader";
import TableRowWithIcon from "./TableRowWithIcon";

export interface TableRowData {
    id: number; // 각 행의 고유 ID
    /** 테이블 data 배열 */
    columns: (string | JSX.Element)[];
}

interface TableRowWithIconProps {
    /** 테이블 title row */
    columns: (string | JSX.Element)[];
    /** 테이블 data row 배열 */
    rows: TableRowData[];
    /** 테이블 셀 width */
    width?: string;
    iconPaths: { [key: number]: string }; // 행별 아이콘 경로
    onIconClick: (id: number) => void; // 아이콘 클릭 시 이벤트
    onRowClick?: (id: number) => void;
}

const TableWithIconAndButton: React.FC<TableRowWithIconProps> = ({
    columns,
    rows,
    width = "auto",
    iconPaths = "정보 변경",
    onIconClick,
    onRowClick = () => { },
}) => {
    return (
        <table className="min-w-full divide-y divide-gray-200">
            <TableHeader columns={columns} width={width} />
            <tbody className="bg-white divide-y divide-gray-100">
                {rows.map((row) => (
                    <TableRowWithIcon
                        key={row.id}
                        columns={row.columns}
                        width={width}
                        iconPath={iconPaths[row.id]}
                        styleName="w-3.5 h-3.5"
                        onRowClick={() => onRowClick?.(row.id)}
                        onIconClick={() => onIconClick?.(row.id)}
                    />
                ))}
            </tbody>
        </table>
    );
};

export default TableWithIconAndButton;
