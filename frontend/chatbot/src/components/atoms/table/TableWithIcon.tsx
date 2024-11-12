import React from "react";
import TableHeader from "./TableHeader";
import TableRowWithIcon from "./TableRowWithIcon";

interface TableRowData {
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
}

const TableWithIconAndButton: React.FC<TableRowWithIconProps> = ({
    columns,
    rows,
    width = "auto",
    iconPaths = "정보 변경",
    onIconClick,
}) => {
    return (
        <table className="min-w-full border-collapse">
            <TableHeader columns={columns} width={width} textColor="#818CF8" />
            <tbody>
                {rows.map((row) => (
                    <TableRowWithIcon
                        key={row.id}
                        columns={row.columns}
                        width={width}
                        iconPath={iconPaths[row.id]} // 행별 아이콘 경로
                        styleName={"w-4 h-4"}
                        onIconClick={() => onIconClick(row.id)} // 특정 행 ID 전달
                    />
                ))}
            </tbody>
        </table>
    );
};

export default TableWithIconAndButton;
