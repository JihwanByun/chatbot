import React from "react";
import ButtonOnlyIcon from "../button/ButtonOnlyIcon";
import { Avatar } from "../avatar/Avatar";

interface TableRowWithAvatarAndButtonProps {
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

const TableRowWithAvatarAndButton: React.FC<
  TableRowWithAvatarAndButtonProps
> = ({
  columns,
  width = "auto",
  textColor = "#000000",
  iconPath,
  styleName,
  onIconClick,
  onRowClick,
}) => {
    return (
      <tr
        className="text-left bg-white hover:bg-gray-50 transition-colors duration-150"
        onClick={onRowClick}
      >
        <td className="pl-5 py-1">
          <Avatar name={columns[1]} size="2rem" />{" "}
          {/* 이름 컬럼을 기반으로 아바타 생성 */}
        </td>
        {columns.map((column, index) => (
          <td
            key={index}
            style={{ width, color: textColor }}
            className="px-2 py-1 truncate"
          >
            {column}
          </td>
        ))}
      </tr>
    );
  };

export default TableRowWithAvatarAndButton;
