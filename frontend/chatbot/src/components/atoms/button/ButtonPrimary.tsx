export type ButtonProps = {
	btnName: string;
	styleName?: string;
	handleClick?: () => void; // 함수 타입 지정
	isDisabled?: boolean;
};

const ButtonPrimary = ({
	btnName,
	styleName = "bg-white-100",
	handleClick,
	isDisabled = false,
}: ButtonProps) => {
	return (
		<div>
			<button
				className={styleName}
				data-ripple-light="true"
				onClick={handleClick} // 클릭 시 함수 호출
				disabled={isDisabled}
			>
				{btnName}
			</button>
		</div>
	);
};

export default ButtonPrimary;
