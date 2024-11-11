import React from "react";
import TableHeader from "./TableHeader";
import TableRowWithIconAndButton from "./TableRowWithIconAndButton";

interface TableRowData {
    /** 테이블 data 배열 */
    columns: (string | JSX.Element)[];
    /** 테이블 data icon 경로 */
    iconPath: string;
}

interface TableWithIconAndButtonProps {
    /** 테이블 title row */
    columns: (string | JSX.Element)[];
    /** 테이블 data row 배열 */
    rows: TableRowData[];
    /** 테이블 셀 width */
    width?: string;
    /** 테이블 data 버튼 라벨 */
    buttonLabel?: string;
    /** 테이블 data 버튼 click handler */
    onButtonClick?: () => void;
}

const TableWithIconAndButton: React.FC<TableWithIconAndButtonProps> = ({
    columns,
    rows,
    width = "auto",
    buttonLabel = "정보 변경",
    onButtonClick,
}) => {
    return (
        <table className="min-w-full border-separate border-spacing-y-1">
            <TableHeader columns={columns} width={width} />
            <tbody>
                {rows.map((row, index) => (
                    <TableRowWithIconAndButton
                        key={index}
                        columns={row.columns}
                        width={width}
                        iconPath={row.iconPath}
                        buttonLabel={buttonLabel}
                        onButtonClick={onButtonClick}
                    />
                ))}
            </tbody>
        </table>
    );
};

export default TableWithIconAndButton;
