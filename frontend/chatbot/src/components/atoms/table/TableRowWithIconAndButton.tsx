import React from "react";
import ButtonPrimary from "../button/ButtonPrimary";

interface TableRowWithIconAndButtonProps {
    /** 컬럼에 입력되는 정보 */
    columns: (string | JSX.Element)[];
    /** 넓이(ex - 16px , 5rem) */
    width?: string;
    /** 텍스트 색상 */
    textColor?: string;
    /** 아이콘 경로 */
    iconPath?: string;
    /** 버튼 라벨 */
    buttonLabel: string;
    /** 버튼 click handler */
    onButtonClick?: () => void;
}

const TableRowWithIconAndButton: React.FC<TableRowWithIconAndButtonProps> = ({
    columns,
    width = "auto",
    textColor = "#718EBF",
    iconPath,
    buttonLabel,
    onButtonClick,
}) => {
    return (
        <tr className="text-left bg-white">
            <td className="rounded-l-2xl pl-2 max-w-[6rem]">
                <img src={iconPath} alt="Sample Icon" width="40rem" />
            </td>
            {columns.map((column, index) => (
                <td
                    key={index}
                    style={{ width, color: textColor }}
                    className="px-4 py-2 max-w-[30rem] truncate"
                >
                    {column}
                </td>
            ))}
            <td
                style={{ width }}
                className="px-2 py-2 rounded-r-2xl max-w-[10rem]"
            >
                {/* <button className='border-indigo-300' onClick={onButtonClick}>{buttonLabel}</button> */}
                <ButtonPrimary
                    btnName={buttonLabel}
                    styleName="border-indigo-400 text-indigo-400 py-1 px-4 rounded-2xl hover:bg-indigo-400 hover:text-white"
                    handleClick={onButtonClick}
                />
            </td>
        </tr>
    );
};

export default TableRowWithIconAndButton;
