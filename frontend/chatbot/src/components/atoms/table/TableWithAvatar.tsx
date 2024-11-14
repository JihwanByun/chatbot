import React from "react";
import MemberTableHeader from "./MemberTableHeader";
import TableRowWithAvatar from "./TableRowWithAvatar";

export interface TableRowData {
  id: number;
  columns: (string | JSX.Element)[];
}

interface TableRowWithAvatarProps {
  columns: (string | JSX.Element)[];
  rows: TableRowData[];
  width?: string;
  iconPaths: { [key: number]: string };
  onIconClick: (id: number) => void;
  onRowClick?: (id: number) => void;
}

const TableWithAvaterAndButton: React.FC<TableRowWithAvatarProps> = ({
  columns,
  rows,
  width = "auto",
  iconPaths = "정보 변경",
  onIconClick,
  onRowClick = () => { },
}) => {
  return (
    <table className="min-w-full divide-y divide-gray-200 shadow-sm rounded-lg overflow-hidden">
      <MemberTableHeader columns={columns} width={width} />
      <tbody className="bg-white divide-y divide-gray-100">
        {rows.map((row) => (
          <TableRowWithAvatar
            key={row.id}
            columns={row.columns}
            width={width}
            iconPath={iconPaths[row.id]}
            styleName="w-2 h-2"
            onRowClick={() => onRowClick?.(row.id)}
            onIconClick={() => onIconClick?.(row.id)}
          />
        ))}
      </tbody>
    </table>
  );
};

export default TableWithAvaterAndButton;