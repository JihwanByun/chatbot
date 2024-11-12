import React from "react";
import ButtonOnlyIcon from "../button/ButtonOnlyIcon";

interface TableRowWithIconAndButtonProps {
  /** 컬럼에 입력되는 정보 */
  columns: string[];
  /** 넓이(ex - 16px , 5rem) */
  width?: string;
  /** 텍스트 색상 */
  textColor?: string;
  /** 아이콘 경로 */
  iconPath: string;
  /** css */
  styleName: string;
  /** 아이콘 click handler */
  onIconClick?: () => void;
  onRowClick?: () => void;
}

const TableRowWithIconAndButton: React.FC<TableRowWithIconAndButtonProps> = ({
  columns,
  width = "auto",
  textColor = "#000000",
  iconPath,
  styleName,
  onIconClick,
  onRowClick,
}) => {
  console.log("*" + columns);

  return (
    <tr className="text-left bg-white" onClick={onRowClick}>
      <td className="pl-5 max-w-[6rem]">
        <ButtonOnlyIcon
          icon={iconPath}
          styleName={styleName}
          onClick={onIconClick}
        />
      </td>
      {columns.map((column, index) => (
        <td
          key={index}
          style={{ width, color: textColor }}
          className="px-4 py-2 max-w-[50rem] truncate"
        >
          {column}
        </td>
      ))}
    </tr>
  );
};

export default TableRowWithIconAndButton;
